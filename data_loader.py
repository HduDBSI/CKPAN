import os
import numpy as np
import logging
import collections
from tqdm import tqdm
from multiprocessing import Pool
import math

def load_data(args):
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

    # path = "layer_{}-cs_{}-uts_{}-its_{}.npz".format(args.n_layer, args.course_size,
    #                                                  args.user_triple_set_size, args.item_triple_set_size)
    path = "{}_part-layer_{}-cs_{}-uts_{}-its_{}-uvs_{}.npz".format(args.dataset, args.n_layer, args.course_size,
                                                                   args.user_triple_set_size, args.item_triple_set_size,
                                                                   args.user_video_size)
    train_eval_test_path = f"{args.dataset}_part_train_eval_test.npz"
    if os.path.exists(train_eval_test_path):
        train_eval_test = np.load(train_eval_test_path, allow_pickle=True)
        train_data, eval_data, test_data, items = train_eval_test['train_data'], train_eval_test['eval_data'], train_eval_test['test_data'], train_eval_test['items']
        user_init_entity_set, item_init_entity_set = train_eval_test['user_init_entity_set'].item(), train_eval_test['item_init_entity_set'].item()
    else:
        train_data, eval_data, test_data, items, user_init_entity_set, item_init_entity_set = load_rating(args)
        np.savez(train_eval_test_path, train_data=train_data, eval_data=eval_data, test_data=test_data, items=items,
                 user_init_entity_set=user_init_entity_set, item_init_entity_set=item_init_entity_set)

    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        n_entity, n_relation = data['n_entity'].item(), data['n_relation'].item()
        args.padding_idx = n_entity
        user_triple_sets, item_triple_sets = data['user_triple_sets'].item(), data['item_triple_sets'].item()
        user_course_videos = data['user_course_videos'].item()
        user_course_lists, item_course_lists = data['user_course_lists'].item(), data['item_course_lists'].item()

        return train_data, eval_data, test_data, items, n_entity, n_relation, user_triple_sets, item_triple_sets, user_course_videos, user_course_lists, item_course_lists


    n_entity, n_relation, kg = load_kg(args)
    args.padding_idx = n_entity

    logging.info("contructing items' kg triple sets ...")
    item_triple_sets, item_course_lists = kg_propagation(args, kg, item_init_entity_set, args.item_triple_set_size, desc='item kg propagation', is_user=False)

    logging.info("contructing users' kg triple sets ...")
    user_triple_sets, user_course_lists = multiprocess(args, kg, user_init_entity_set, args.user_triple_set_size, desc='user kg propagation')

    user_course_videos = load_course_videos(args, user_course_lists)

    np.savez(path,
             n_entity=n_entity, n_relation=n_relation,
             user_triple_sets=user_triple_sets, item_triple_sets=item_triple_sets,
             user_course_videos=user_course_videos,
             user_course_lists=user_course_lists, item_course_lists=item_course_lists)
    return train_data, eval_data, test_data, items, n_entity, n_relation, user_triple_sets, item_triple_sets, user_course_videos, user_course_lists, item_course_lists


def load_rating(args):
    rating_file = os.path.join(args.data_dir_processed, args.dataset, 'user-course_filtered_final')
    # if os.path.exists(rating_file + '.npy'):
    #     rating_np = np.load(rating_file + '.npy')
    # else:
    rating_np = np.loadtxt(rating_file, dtype=np.int32)
        # np.save(rating_file + '.npy', rating_np)
    return dataset_split(rating_np)


def dataset_split(rating_np):
    eval_ratio = 0.2
    test_ratio = 0.2

    n_rating = len(rating_np)

    eval_indices = np.random.choice(n_rating, size=int(n_rating * eval_ratio), replace=False)
    left = set(range(n_rating)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_rating * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    user_history_item_dict, item_neighbor_item_dict = collaboration_propagation(rating_np, train_indices)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_item_dict.keys()]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_item_dict.keys()]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_item_dict.keys()]
    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]
    items = list(set(rating_np[:, 1]))

    return train_data, eval_data, test_data, items, user_history_item_dict, item_neighbor_item_dict


def collaboration_propagation(rating_np, train_indices):
    logging.info("contructing users' initial entity set ...")
    user_history_item_dict = dict()
    item_history_user_dict = dict()
    item_neighbor_item_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        if user not in user_history_item_dict:
            user_history_item_dict[user] = []
        user_history_item_dict[user].append(item)
        if item not in item_history_user_dict:
            item_history_user_dict[item] = []
        item_history_user_dict[item].append(user)

    logging.info("contructing items' initial entity set ...")
    item_sim = dict()
    for item_i in item_history_user_dict.keys():
        if item_i not in item_sim:
            item_sim[item_i] = dict()
        for item_j in item_history_user_dict.keys():
            if item_j not in item_sim:
                item_sim[item_j] = dict()
            if item_i == item_j or item_j in item_sim[item_i]:
                continue
            N_i = set(item_history_user_dict[item_i])
            N_j = set(item_history_user_dict[item_j])
            sim_ij = len(N_i & N_j) / len(N_i | N_j)
            sim_ji = sim_ij
            item_sim[item_i][item_j] = sim_ij
            item_sim[item_j][item_i] = sim_ji

    for item_i in item_sim.keys():
        filtered = dict(filter(lambda x: x[1] > 0, item_sim[item_i].items()))
        item_neighbor_item_dict[item_i] = [item_j for item_j, value in list(sorted(filtered.items(), key=lambda x:x[1], reverse=True))]

    item_list = set(rating_np[:, 1])
    for item in item_list:
        if item not in item_neighbor_item_dict or len(item_neighbor_item_dict[item]) == 0:
            item_neighbor_item_dict[item] = [item]
    return user_history_item_dict, item_neighbor_item_dict


def load_kg(args):
    kg_file = os.path.join(args.data_dir_processed, args.dataset, 'kg_final')
    logging.info("loading kg file: %s.npy", kg_file)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file, dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    return n_entity, n_relation, kg


def construct_kg(kg_np):
    logging.info("constructing knowledge graph ...")
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def kg_propagation(args, kg, init_entity_dict, set_size, desc='user kg propagation', is_user=True):
    n_layer = args.n_layer
    course_size = args.course_size

    triple_sets = dict()
    course_lists = dict()

    for obj in tqdm(init_entity_dict.keys(), ncols=100, desc=desc):

        if obj not in triple_sets.keys():
            triple_sets[obj] = []

        if is_user:
            course_list = list(set(init_entity_dict[obj]))
            course_list = np.random.choice(course_list, size=course_size, replace=(len(course_list) < course_size))
        else:
            course_list = list(set(init_entity_dict[obj]) - {obj})
            if len(course_list) < 1:
                course_list = [obj] * course_size
            elif len(course_list) < (course_size - 1):
                tmp_course_list = [obj]
                for i in range(course_size-1):
                    tmp_course_list.append(course_list[i % len(course_list)])
                course_list = tmp_course_list
            else:
                course_list = [obj] + course_list[:(course_size - 1)]

        course_lists[obj] = course_list

        for l in range(n_layer):
            layer_i = []
            for tmp_index, course_index in enumerate(course_list):
                h, r, t = [], [], []
                if l == 0:
                    entities = [course_index]
                else:
                    entities = triple_sets[obj][-1][tmp_index][2]

                layer_i_tmp_index = 0
                layer_i_relation_dict = dict()
                for entity in entities:
                    for tail_and_relation in kg[entity]:
                        h.append(entity)
                        t.append(tail_and_relation[0])
                        r.append(tail_and_relation[1])
                        tmp_r = tail_and_relation[1]
                        if tmp_r not in layer_i_relation_dict:
                            layer_i_relation_dict[tmp_r] = []
                        layer_i_relation_dict[tmp_r].append(layer_i_tmp_index)
                        layer_i_tmp_index += 1
                if len(h) == 0:
                    triple_sets[obj].append(triple_sets[obj][-1])
                else:
                    layer_i_relation_keys = list(layer_i_relation_dict.keys())
                    choose_num_dict = dict()
                    num_dict = dict()
                    for key in layer_i_relation_keys:
                        choose_num_dict[key] = 0
                        num_dict[key] = len(layer_i_relation_dict[key])
                    choose_num = 0
                    full_selected = set()
                    while choose_num < set_size:
                        for key in layer_i_relation_keys:
                            if choose_num_dict[key] < num_dict[key]:
                                choose_num_dict[key] = choose_num_dict[key] + 1
                                choose_num += 1
                                if choose_num >= set_size:
                                    break
                            else:
                                full_selected.add(key)
                        if choose_num >= set_size:
                            break
                        if len(full_selected) == len(layer_i_relation_keys):
                            for key in layer_i_relation_keys:
                                choose_num_dict[key] = choose_num_dict[key] + 1
                                choose_num += 1
                                if choose_num >= set_size:
                                    break

                    indices = None
                    for r_key, cnt in choose_num_dict.items():
                        layer_i_tmp_index_list = layer_i_relation_dict[r_key]
                        if indices is None:
                            tmp_indices = np.array(layer_i_tmp_index_list)
                            if len(tmp_indices) < cnt:
                                indices = np.concatenate((tmp_indices, np.random.choice(layer_i_tmp_index_list, size=(cnt-len(tmp_indices)), replace=True)))
                            else:
                                indices = np.random.choice(layer_i_tmp_index_list, size=cnt, replace=False)
                        else:
                            tmp_indices = np.array(layer_i_tmp_index_list)
                            if len(tmp_indices) < cnt:
                                indices = np.concatenate((indices, tmp_indices, np.random.choice(layer_i_tmp_index_list, size=(cnt-len(tmp_indices)), replace=True)))
                            else:
                                indices = np.concatenate((indices, np.random.choice(layer_i_tmp_index_list, size=cnt, replace=False)))

                    h = [h[i] for i in indices]
                    r = [r[i] for i in indices]
                    t = [t[i] for i in indices]
                layer_i.append([h, r, t])
            triple_sets[obj].append(layer_i)

    # {users: [layer_i[course_k[entities]], ...]}, {users: [course_k]}
    return triple_sets, course_lists


def multiprocess(args, kg, init_entity_dict, set_size, desc):
    process_list = []
    pool = Pool(args.n_worker)
    tmp_batch_size = math.ceil(len(init_entity_dict) / args.n_worker)
    start_index = 0
    keys = list(init_entity_dict.keys())
    sub_init_entity_dict = []
    for i in range(args.n_worker):
        tmp_init_entity_dict = dict()
        for k in keys[start_index: start_index + tmp_batch_size]:
            tmp_init_entity_dict[k] = init_entity_dict[k]
        start_index += tmp_batch_size
        sub_init_entity_dict.append(tmp_init_entity_dict)
    for i in range(args.n_worker):
        process_list.append(pool.apply_async(kg_propagation, args=(args, kg, sub_init_entity_dict[i], set_size, desc)))
    pool.close()
    pool.join()

    triple_sets = dict()
    course_lists = dict()

    for p in process_list:
        _triple_sets, _course_lists = p.get()
        triple_sets.update(_triple_sets)
        course_lists.update(_course_lists)
    pool.terminate()
    return triple_sets, course_lists


def load_course_videos(args, course_lists):
    user_video_size = args.user_video_size
    padding_idx = args.padding_idx
    user_course_videos_path = os.path.join(args.data_dir_processed, args.dataset, 'user-video.csv')
    entity_path = os.path.join(args.data_dir_processed, args.dataset, 'entity.csv')

    entity_map = dict()
    with open(entity_path, 'r') as reader:
        lines = reader.readlines()
        for line in lines[1:]:
            line = line.strip().split('\t')
            entity_id, entity_index = line[0], line[1]
            entity_map[entity_id] = int(entity_index)

    user_course_videos = dict()
    with open(user_course_videos_path, 'r') as reader:
        lines = reader.readlines()
        for line in lines[1:]:
            line = line.strip().split('\t')
            user_index, video_id, course_id = int(line[0]), int(line[1]), int(line[2])
            video_index = video_id
            course_index = course_id
            if user_index not in course_lists.keys():
                continue
            if user_index not in user_course_videos.keys():
                user_course_videos[user_index] = dict()
            if course_index not in course_lists[user_index]:
                continue
            if course_index not in user_course_videos[user_index].keys():
                user_course_videos[user_index][course_index] = set()
            user_course_videos[user_index][course_index].add(video_index)

    users = user_course_videos.keys()
    for user_index in users:
        for course_index, all_videos in user_course_videos[user_index].items():
            all_videos = list(all_videos)
            replace = (len(all_videos) < user_video_size)
            if replace:
                all_videos = all_videos + [padding_idx] * (user_video_size - len(all_videos))
            videos = np.random.choice(all_videos, size=user_video_size, replace=False)
            user_course_videos[user_index][course_index] = videos

    final_user_course_videos = dict()
    for user_index, courses in course_lists.items():
        final_user_course_videos[user_index] = []
        for course_index in courses:
            if user_index not in user_course_videos.keys() or course_index not in user_course_videos[user_index].keys():
                final_user_course_videos[user_index].append(np.array([padding_idx] * user_video_size))
            else:
                final_user_course_videos[user_index].append(user_course_videos[user_index][course_index])

    return final_user_course_videos

