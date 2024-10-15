import torch
import torch.nn as nn
import numpy as np

class CVRS(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(CVRS, self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.device = args.device
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.course_size = args.course_size
        self.user_video_size = args.user_video_size
        self.user_triple_set_size = args.user_triple_set_size
        self.item_triple_set_size = args.item_triple_set_size
        self.padding_idx = self.n_entity
        self.dropout_rate = args.dropout_rate
        self.entity_emb = nn.Embedding(self.n_entity + 1, self.dim, padding_idx=self.padding_idx)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.layer_emb = nn.Embedding(self.n_layer, self.dim)
        self.W = nn.Parameter(torch.empty(self.dim * 2, self.dim))
        self.W_ht = nn.Parameter(torch.empty(self.dim * 2, self.dim))
        self.W_course = nn.Parameter(torch.empty(self.dim, self.dim))
        self.W_video = nn.Parameter(torch.empty(self.dim, self.dim))
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.layer_emb.weight)
        with torch.no_grad():
            torch.fill_(self.entity_emb.weight[self.padding_idx], 0)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.W_ht)
        nn.init.xavier_uniform_(self.W_course)
        nn.init.xavier_uniform_(self.W_video)

    def forward(self, user_course_videos, items, neg_items, user_triple_set, item_triple_set, neg_item_triple_set, user_course_list, item_course_list, neg_item_course_list):
        u = self.run_user(user_course_videos, user_triple_set, user_course_list)
        c = self.run_item(items, item_triple_set, item_course_list)
        n_c = self.run_item(neg_items, neg_item_triple_set, neg_item_course_list)
        scores = self.score(u, c)
        neg_scores = self.score(u, n_c)
        return scores, neg_scores, 0

    def run_user(self, user_course_videos_watched, user_triple_set, user_course_lists):
        user_course_videos_central_emb = self._extract_central_emb_via_video(user_course_videos_watched)

        user_course_central_emb = self._extract_central_emb_via_course(user_course_lists, user_course_videos_central_emb)
        user_course_videos_central_emb = torch.sigmoid(torch.matmul(user_course_videos_central_emb.mean(dim=-2), self.W_video))

        u = self._extract_layer_i_emb(user_triple_set, True, user_course_central_emb, self.user_triple_set_size, user_course_videos_central_emb)

        u = u.permute(0, 2, 1, 3)
        u = self.dropout(u)
        u = nn.functional.normalize(u, dim=-1)

        u = u.sum(dim=2).mean(dim=1)

        return u

    def run_item(self, items, item_triple_set, item_course_lists):
        item_cf_course_lists = item_course_lists[:, 1:]
        item_course_central_emb = self.entity_emb(items) + self.entity_emb(items) * self._extract_central_emb_via_course(item_cf_course_lists)

        c = self._extract_layer_i_emb(item_triple_set, False, item_course_central_emb, self.item_triple_set_size)

        c = c.permute(0, 2, 1, 3)

        c = self.dropout(c)

        c = nn.functional.normalize(c, dim=-1)

        c = c.sum(dim=2)

        cw = torch.sigmoid(torch.matmul(c[:, 1:, :], c[:, :1, :].permute(0, 2, 1)))

        p_cw = torch.softmax(cw, dim=-2)

        mean_cf = (p_cw * c[:, 1:, :]).sum(dim=1)

        alpha_c = torch.sigmoid((self.entity_emb(items) * mean_cf).sum(dim=-1, keepdim=True))
        c = (alpha_c / (1 + alpha_c)) * mean_cf + c[:, :1, :].squeeze(dim=1) + self.entity_emb(items)

        return c

    def _extract_central_emb_via_course(self, obj_course_lists, user_course_videos_central_emb=None):
        '''

        :param user_course_lists: [user, course, dim]
        :param user_course_videos_watched: [user, course, video, dim]
        :return:
        '''
        obj_course_lists_emb = self.entity_emb(obj_course_lists)
        if user_course_videos_central_emb is not None:
            user_course_videos_central_emb = torch.sigmoid(torch.matmul(user_course_videos_central_emb, self.W_course))
            obj_course_lists_emb = obj_course_lists_emb + user_course_videos_central_emb

        obj_course_lists_emb = obj_course_lists_emb.mean(dim=-2)

        return obj_course_lists_emb


    def _extract_central_emb_via_video(self, user_course_videos_watched):
        # [user, course_size, video_size, dim]
        user_course_videos_watched_emb = self.entity_emb(user_course_videos_watched)
        user_course_videos_central_emb = user_course_videos_watched_emb.sum(dim=-2)
        return user_course_videos_central_emb

    def _extract_layer_i_emb(self, triple_set, is_user, course_central_emb, set_size, user_course_videos_central_emb=None):
        h_emb = self.entity_emb(triple_set[:, :, :, 0, :])
        r_emb = self.relation_emb(triple_set[:, :, :, 1, :])
        t_emb = self.entity_emb(triple_set[:, :, :, 2, :])

        hr = torch.cat((h_emb, r_emb), dim=-1)
        hr_W = torch.matmul(hr, self.W)

        p = torch.matmul(hr_W.reshape(-1, set_size * self.course_size * self.n_layer, self.dim), course_central_emb.unsqueeze(dim=2))
        p = torch.softmax(torch.sigmoid(p.reshape(-1, self.n_layer, self.course_size, set_size)), dim=-1).unsqueeze(dim=-1)

        o = p * t_emb
        if is_user:
            ht = torch.cat((h_emb, t_emb), dim=-1)
            ht_W = torch.matmul(ht, self.W_ht)

            p2 = torch.matmul(ht_W.reshape(-1, set_size * self.course_size * self.n_layer, self.dim), user_course_videos_central_emb.unsqueeze(dim=2))
            p2 = torch.sigmoid(p2.reshape(-1, self.n_layer, self.course_size, set_size)).unsqueeze(dim=-1)
            o = p2 * o

        o = o.sum(dim=-2)

        return o

    def score(self, user_embeddings, item_embeddings):
        o = user_embeddings * item_embeddings
        o = o.sum(dim=1)
        return o


    def predict(self, user_course_videos, items, user_triple_set, item_triple_set, user_course_list, item_course_list):
        items = items.squeeze(dim=0)
        item_triple_set = item_triple_set.squeeze(dim=0)
        item_course_list = item_course_list.squeeze(dim=0)
        u = self.run_user(user_course_videos, user_triple_set, user_course_list)
        c = self.run_item(items, item_triple_set, item_course_list)
        scores = self.score(u, c)
        return scores, 0

