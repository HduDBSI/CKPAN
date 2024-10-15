import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import numpy as np
from tqdm import tqdm, trange
import os
import logging

from model import CVRS

formmater = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


def train(args, data_info):
    log_output_file_path = os.path.join(args.checkpoint_dir, 'train.log')

    train_logger = logging.getLogger("train_logger")
    train_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_output_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formmater)
    train_logger.addHandler(file_handler)
    train_logger.propagate = False
    train_logger.info(args)
    print(args)

    train_dataloader, eval_dataloader, test_dataloader, n_entity, n_relation = get_dataloader(args, data_info)

    model, optimizer, loss_func, device = _init_model(args, n_entity, n_relation)

    model.train()
    eval_max_score = 0
    for epoch in range(args.n_epoch):
        loop = tqdm(train_dataloader, desc='epoch: {}/{}'.format(epoch, args.n_epoch), ncols=100)
        epoch_rec_loss = 0.
        epoch_kg_loss = 0.
        n_batch = len(train_dataloader)
        for batch_index, batch in enumerate(loop, start=1):
            batch = trans_batch_to_device(args, batch, device)
            items = batch[1]
            neg_items = batch[2]
            # pos_label = torch.ones(items.shape, device=args.device)
            # neg_label = torch.zeros(neg_items.shape, device=args.device)
            pos_label = torch.ones(items.shape).to(device)
            neg_label = torch.zeros(neg_items.shape).to(device)
            score, neg_score, kg_loss = model(*batch)
            # kg_loss = -1 * kg_loss
            rec_loss = loss_func(score, pos_label) + loss_func(neg_score, neg_label)
            # rec_loss = -1 * torch.mean(nn.LogSigmoid()(score-neg_score))
            # loss = args.gamma1 * rec_loss + (1 - args.gamma1) * kg_loss
            # loss = rec_loss + (1 - args.gamma1) * kg_loss
            loss = rec_loss
            # score, neg_score, kg_loss = model(user_course_videos, course_videos, items, neg_items, user_triple_set, item_triple_set, neg_item_triple_set, user_course_list, item_course_list, neg_item_course_list)
            optimizer.zero_grad()

            # if args.use_cuda:
            #     pos_label = pos_label.cuda()
            #     neg_label = neg_label.cuda()

            loss.backward()
            optimizer.step()
            rec_loss = rec_loss.item()
            # kg_loss = kg_loss.item()
            epoch_rec_loss += rec_loss
            epoch_kg_loss += kg_loss
            loop.set_postfix(rec_loss="{:.6f}".format(rec_loss), kg_loss='{:.6f}'.format(kg_loss))
            train_logger.info("train epoch: {}/{}, batch: {}, rec_loss: {:.6f}, kg_loss: {:.6f}".format(epoch, args.n_epoch, batch_index, rec_loss, kg_loss))
        train_logger.info("train epoch: {}/{}, mean_rec_loss: {:.6f}, mean_kg_loss: {:.6f}".format(epoch, args.n_epoch, epoch_rec_loss / n_batch, epoch_kg_loss / n_batch))

        # dist.barrier()

        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                K = [1, 5, 10, 20, 50]
                R = {k: [] for k in K}
                N = {k: [] for k in K}
                # user_limited_count = len(eval_dataloader)
                user_limited_count = 100
                # if user_limited_count > 500:
                #     user_limited_count = 500
                eval_loop = tqdm(total=user_limited_count, ncols=100)
                user_count = 0
                total_kg_loss = 0.
                total_rec_loss = 0.
                for batch_index, eval_batch in enumerate(eval_dataloader, start=1):
                    user_count += 1
                    if user_count > user_limited_count:
                        break
                    pos_count = eval_batch[-1].cpu().item()
                    batch = trans_batch_to_device(args, eval_batch[:-1], device)
                    scores, kg_loss = model.predict(*batch)
                    # print(f"scores: {scores}")

                    # to address OOM
                    # user_course_videos, item_index_list, user_triple_set, item_triple_set_list, user_course_list, item_course_list_list = eval_batch[0], eval_batch[1], eval_batch[2], eval_batch[3], eval_batch[4], eval_batch[5]
                    # scores, kg_loss = split(user_course_videos, item_index_list, user_triple_set, item_triple_set_list, user_course_list, item_course_list_list, model, device)

                    # kg_loss = -1 * kg_loss
                    # kg_loss = kg_loss.item()
                    total_kg_loss += kg_loss
                    # labels = torch.zeros(scores.shape, device=args.device)
                    labels = torch.zeros(scores.shape).to(device)
                    for i in range(pos_count):
                        labels[i] = 1
                    rec_loss = loss_func(scores, labels).item()
                    total_rec_loss += rec_loss
                    topk(-scores, pos_count, K, R, N)
                    eval_loop.update(1)
                    train_logger.info("eval batch {}/{} rec_loss:{:.6f}, kg_loss:{:.6f}".format(batch_index, user_limited_count, rec_loss, kg_loss))
                mean_rec_loss = total_rec_loss / user_limited_count
                mean_kg_loss = total_kg_loss / user_limited_count
                train_logger.info("eval epoch {}/{} mean_rec_loss:{:.6f}, mean_kg_loss:{:.6f}".format(epoch, args.n_epoch, mean_rec_loss, mean_kg_loss))
                target_score = show_topk(train_logger, K, R, N, desc='eval')
                if target_score > eval_max_score:
                    # print("current score: {:.6f}  >  max score: {:.6f}. Save to best.state_dict.".format(target_score, eval_max_score))
                    train_logger.info("current score: {:.6f}  >  max score: {:.6f}. Save to best.state_dict.".format(target_score, eval_max_score))
                    eval_max_score = target_score
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best.state_dict"))
                # mean_loss = args.gamma1 * mean_rec_loss + (1-args.gamma1) * mean_kg_loss
                # if mean_rec_loss < eval_loss:
                #     train_logger.info("mean_rec_loss: {:.6f} < eval_loss: {:.6f}. Save to best.state_dict.".format(mean_rec_loss, eval_loss))
                #     eval_loss = mean_rec_loss
                #     torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best.state_dict"))

                model.train()

    # dist.destroy_process_group()


def _init_model(args, n_entity, n_relation):
    device = torch.device(args.device)

    model = CVRS(args, n_entity, n_relation)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.l2_weight,
    )
    loss_func = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_func, device


class TrainDataset(Dataset):
    def __init__(self, args, dataset, user_triple_sets, item_triple_sets, user_course_videos, user_course_lists, item_course_lists, items):
        self.dataset = dataset
        self.user_triple_sets = user_triple_sets
        self.item_triple_sets = item_triple_sets
        self.user_course_videos = user_course_videos
        self.user_course_lists = user_course_lists
        self.item_course_lists = item_course_lists
        self.items = items
        self.padding_idx = args.padding_idx
        self.user_acted = dict()
        for row in self.dataset:
            user_index = row[0]
            item_index = row[1]
            if user_index not in self.user_acted:
                self.user_acted[user_index] = [self.padding_idx]
            self.user_acted[user_index].append(item_index)
        # for user_index in user_course_lists:
        #     if user_index not in self.course_videos:
        #         self.course_videos[user_index] = []
        #     for course_index in user_course_lists[user_index]:
        #         self.course_videos[user_index].append(course_videos[course_index])
        self.neg_items = dict()
        for user_index, items_index in self.user_acted.items():
            for item_index in items_index:
                neg_item_index = np.random.choice(self.items, size=1)[0]
                while neg_item_index in self.user_acted[user_index]:
                    neg_item_index = np.random.choice(self.items, size=1)[0]
        for row_index, row in enumerate(self.dataset):
            user_index = row[0]
            neg_item_index = np.random.choice(self.items, size=1)[0]
            while neg_item_index in self.user_acted[user_index]:
                neg_item_index = np.random.choice(self.items, size=1)[0]
            self.neg_items[row_index] = neg_item_index



    def __getitem__(self, index):
        user_index = self.dataset[index, 0]
        item_index = self.dataset[index, 1]
        user_triple_set = self.user_triple_sets[user_index]
        item_triple_set = self.item_triple_sets[item_index]
        user_course_videos = self.user_course_videos[user_index]
        user_course_list = self.user_course_lists[user_index]
        item_course_list = self.item_course_lists[item_index]

        neg_item_index = self.neg_items[index]
        neg_item_triple_set = self.item_triple_sets[neg_item_index]
        neg_item_course_list = self.item_course_lists[neg_item_index]
        # 转化成np.array的原因。注意dataloader的collate_fn属性的默认实现对于list, np.array, scalar的处理方式。
        # return np.array(user_course_videos, dtype=np.int32), np.array(course_videos, dtype=np.int32), item_index, neg_item_index, \
        #        np.array(user_triple_set, dtype=np.int32), np.array(item_triple_set, dtype=np.int32), np.array(neg_item_triple_set, dtype=np.int32), np.array(user_course_list, dtype=np.int32), \
        #        np.array(item_course_list, dtype=np.int32), np.array(neg_item_course_list, dtype=np.int32)
        return np.array(user_course_videos, dtype=np.int32), item_index, neg_item_index, \
               np.array(user_triple_set, dtype=np.int32), np.array(item_triple_set, dtype=np.int32), np.array(neg_item_triple_set, dtype=np.int32), np.array(user_course_list, dtype=np.int32), \
               np.array(item_course_list, dtype=np.int32), np.array(neg_item_course_list, dtype=np.int32)

    def __len__(self):
        return self.dataset.shape[0]


class EvalDataset(Dataset):
    def __init__(self, args, dataset, user_triple_sets, item_triple_sets, user_course_videos, user_course_lists, item_course_lists, items, train_data):
        self.dataset = dataset
        self.user_triple_sets = user_triple_sets
        self.item_triple_sets = item_triple_sets
        self.user_course_videos = user_course_videos
        self.user_course_lists = user_course_lists
        self.item_course_lists = item_course_lists
        self.items = items  # list
        self.padding_idx = args.padding_idx
        self.items_set = set(items)
        self.user_acted = dict()
        self.user_eval_pos = dict()
        for row in train_data:
            user_index = row[0]
            item_index = row[1]
            if user_index not in self.user_acted:
                self.user_acted[user_index] = [self.padding_idx]
            self.user_acted[user_index].append(item_index)
        # for user_index in user_course_lists:
        #     if user_index not in self.course_videos:
        #         self.course_videos[user_index] = []
        #     for course_index in user_course_lists[user_index]:
        #         self.course_videos[user_index].append(course_videos[course_index])
        for row in self.dataset:
            user_index = row[0]
            item_index = row[1]
            if user_index not in self.user_eval_pos:
                self.user_eval_pos[user_index] = []
            self.user_eval_pos[user_index].append(item_index)
        self.user_list = list(set(self.user_acted.keys()) & set(self.user_eval_pos.keys()))
        # if is_test:
        #     self.user_list = list(np.loadtxt(f"{args.dataset}_test_user_{test_num}.txt", dtype=np.int32))

    def __getitem__(self, index):
        user_index = self.user_list[index]
        user_triple_set = self.user_triple_sets[user_index]
        user_course_videos = self.user_course_videos[user_index]
        # course_videos = self.course_videos[user_index]
        user_course_list = self.user_course_lists[user_index]

        item_index_list = []
        item_triple_set_list = []
        item_course_list_list = []
        pos_count = 0
        for item_index in self.user_eval_pos[user_index]:
            item_index_list.append(item_index)
            item_triple_set_list.append(self.item_triple_sets[item_index])
            item_course_list_list.append(self.item_course_lists[item_index])
            pos_count = pos_count + 1
        # if self.is_test:
        candidate_items = list(self.items_set - set(self.user_acted[user_index]) - set(self.user_eval_pos[user_index]))
        # shuffle for auc
        candidate_items_size = len(candidate_items)
        # if candidate_items_size > 700:
        #     candidate_items_size = 700
        candidate_items = np.random.choice(candidate_items, size=candidate_items_size, replace=False)
        # else:
        #     candidate_items = np.random.choice(list(self.items_set - set(self.user_acted[user_index]) - set(self.user_eval_pos[user_index])), size=pos_count, replace=False)
        # if len(candidate_items) > 1000:
        #     candidate_items = np.random.choice(candidate_items, size=1000, replace=False)

        for item_index in candidate_items:
            item_index_list.append(item_index)
            item_triple_set_list.append(self.item_triple_sets[item_index])
            item_course_list_list.append(self.item_course_lists[item_index])

        # return np.array(user_course_videos, dtype=np.int32), np.array(course_videos, dtype=np.int32), np.array(item_index_list, dtype=np.int32), \
        #        np.array(user_triple_set, dtype=np.int32), np.array(item_triple_set_list, dtype=np.int32), \
        #        np.array(user_course_list, dtype=np.int32), np.array(item_course_list_list, dtype=np.int32), pos_count
        return np.array(user_course_videos, dtype=np.int32), np.array(item_index_list, dtype=np.int32), \
               np.array(user_triple_set, dtype=np.int32), np.array(item_triple_set_list, dtype=np.int32), \
               np.array(user_course_list, dtype=np.int32), np.array(item_course_list_list, dtype=np.int32), pos_count

    def __len__(self):
        return len(self.user_list)


def topk(scores, pos_num, K, R, N):
    ranks = scores.argsort().argsort()
    relevance = np.ones(pos_num, dtype=np.float32)
    for k in K:
        hit_count = 0
        rank_relevance = np.asarray([0] * k, dtype=np.float32)
        for i in range(pos_num):
            if ranks[i] < k:
                hit_count = hit_count + 1
                rank_relevance[ranks[i]] = 1
        # Recall
        r = hit_count / pos_num
        R[k].append(r)
        # NDCG
        idcg = getDCG(relevance)
        dcg = getDCG(rank_relevance)
        N[k].append(dcg / idcg)


def getDCG(rel):
    dcg = np.sum(np.divide(np.power(2., rel) - 1, np.log2(np.arange(rel.shape[0], dtype=np.float32) + 2)), dtype=np.float32)
    return dcg


def show_topk(logger, K, R, N, target_metric='N', target_k=10, desc='eval'):
    target_score = None
    for k in K:
        R[k] = np.array(R[k]).mean()
        N[k] = np.array(N[k]).mean()
        if k == target_k:
            if target_metric == 'N':
                target_score = N[k]
            else:
                target_score = R[k]

    assert target_score is not None

    H_print = desc + ' ' + "R@{}:{:.4f} ".format(K[0], R[K[0]])
    for k in K[1:]:
        H_print += "R@{}:{:.4f} ".format(k, R[k])

    N_print = desc + ' ' + "N@{}:{:.4f} ".format(K[0], N[K[0]])
    for k in K[1:]:
        N_print += "N@{}:{:.4f} ".format(k, N[k])

    print()
    print(H_print)
    print(N_print)
    logger.info(H_print)
    logger.info(N_print)
    return target_score


def save_checkpoint(checkpoint_dir, filename, model, optimizer, epoch):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))


def trans_batch_to_device(args, input_batch, device):
    output_batch = []
    for x in input_batch:
        # x = x.to(args.device)
        x = x.to(device)
        output_batch.append(x)
    return output_batch


def get_dataloader(args, data_info):
    train_data, eval_data, test_data, items, n_entity, n_relation, user_triple_sets, item_triple_sets, user_course_videos, user_course_lists, item_course_lists = data_info
    eval_dataset = EvalDataset(args, eval_data, user_triple_sets, item_triple_sets, user_course_videos,
                               user_course_lists, item_course_lists, items, train_data)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1, pin_memory=True,
                                 shuffle=True, num_workers=args.n_worker)
    test_dataset = EvalDataset(args, test_data, user_triple_sets, item_triple_sets, user_course_videos,
                               user_course_lists, item_course_lists, items, train_data)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                                 shuffle=True,  num_workers=args.n_worker)
    train_dataset = TrainDataset(args, train_data, user_triple_sets, item_triple_sets, user_course_videos,
                                 user_course_lists, item_course_lists, items)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.n_worker)
    return train_dataloader, eval_dataloader, test_dataloader, n_entity, n_relation


def test(args, data_info):
    train_dataloader, eval_dataloader, test_dataloader, n_entity, n_relation = get_dataloader(args, data_info)

    log_output_file_path = os.path.join(args.checkpoint_dir, 'test.log')

    test_logger = logging.getLogger("test_logger")
    test_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_output_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formmater)
    test_logger.addHandler(file_handler)
    test_logger.propagate = False
    test_logger.info(args)
    print(args)

    model, _, loss_func, device = _init_model(args, n_entity, n_relation)
    model.load_state_dict(torch.load(args.state_dict_path))
    model.eval()
    with torch.no_grad():
        K = [1, 5, 10, 20, 50]
        R = {k: [] for k in K}
        N = {k: [] for k in K}
        user_limited_count = len(test_dataloader)
        # user_limited_count = 1000
        eval_loop = tqdm(total=user_limited_count, ncols=100)
        user_count = 0
        total_rec_loss = 0.
        total_kg_loss = 0.
        auc_list = []
        f1_list = []
        for batch_index, eval_batch in enumerate(test_dataloader, start=1):
            user_count += 1
            if user_count > user_limited_count:
                break
            pos_count = eval_batch[-1].cpu().item()
            scores, kg_loss = model.predict(*trans_batch_to_device(args, eval_batch[:-1], device))
            # scores, kg_loss = model.module.predict(*trans_batch_to_device(args, eval_batch[:-1], device))

            # to address OOM
            # user_course_videos, item_index_list, user_triple_set, item_triple_set_list, user_course_list, item_course_list_list = \
            # eval_batch[0], eval_batch[1], eval_batch[2], eval_batch[3], eval_batch[4], eval_batch[5]
            # scores, kg_loss = split(user_course_videos, item_index_list, user_triple_set, item_triple_set_list, user_course_list, item_course_list_list, model, device)

            # loss 自带 logits
            scores = scores.sigmoid()
            #
            # kg_loss = -kg_loss.item()
            total_kg_loss += kg_loss
            # labels = torch.zeros(scores.shape, device=args.device)
            labels = torch.zeros(scores.shape).to(device)
            auc_labels = [0] * (2*pos_count)
            for i in range(pos_count):
                labels[i] = 1
                auc_labels[i] = 1
            rec_loss = loss_func(scores, labels).item()
            total_rec_loss += rec_loss
            topk(-scores, pos_count, K, R, N)
            auc_scores = scores[:(2*pos_count)].detach().cpu().tolist()
            auc = roc_auc_score(y_true=auc_labels, y_score=auc_scores)
            predictions = [1 if i >= 0.5 else 0 for i in auc_scores]
            f1 = f1_score(y_true=auc_labels, y_pred=predictions)
            auc_list.append(auc)
            f1_list.append(f1)
            eval_loop.update(1)
            test_logger.info("{}/{} rec_loss:{:.6f}, kg_loss:{:.6f}".format(batch_index, user_limited_count, rec_loss, kg_loss))
        test_logger.info("{}/{} mean_rec_loss:{:.6f}, mean_kg_loss:{:.6f}".format(batch_index, user_limited_count, total_rec_loss / user_limited_count, total_kg_loss / user_limited_count))
        show_topk(test_logger, K, R, N, desc='test')
        auc = float(np.mean(auc_list))
        f1 = float(np.mean(f1_list))
        print(f"auc:{auc:.4f}, f1:{f1:.4f}")
        test_logger.info(f"auc:{auc:.4f}, f1:{f1:.4f}")

def split(user_course_videos, item_index_list, user_triple_set, item_triple_set_list, user_course_list, item_course_list_list, model, device):
    batch = 200
    length = len(item_index_list[0, :])
    batches = length // batch
    start = 0
    scores = []
    user_course_videos = user_course_videos.to(device)
    user_triple_set = user_triple_set.to(device)
    user_course_list = user_course_list.to(device)
    for i in range(batches):
        item_index_list_batch = item_index_list[:1, start: start + batch].to(device)
        item_triple_set_list_batch = item_triple_set_list[:1, start: start + batch].to(device)
        item_course_list_list_batch = item_course_list_list[:1, start: start + batch].to(device)

        scores_batch, _ = model.predict(user_course_videos, item_index_list_batch, user_triple_set, item_triple_set_list_batch, user_course_list, item_course_list_list_batch)
        # print(f"scores_batch: {scores_batch}")
        start = start + batch
        scores.append(scores_batch)
    if start < length:
        item_index_list_batch = item_index_list[:1, start: start + batch].to(device)
        item_triple_set_list_batch = item_triple_set_list[:1, start: start + batch].to(device)
        item_course_list_list_batch = item_course_list_list[:1, start: start + batch].to(device)

        scores_batch, _ = model.predict(user_course_videos, item_index_list_batch, user_triple_set, item_triple_set_list_batch, user_course_list, item_course_list_list_batch)
        start = start + batch
        scores.append(scores_batch)
    scores = torch.concat(scores)
    return scores, 0




