import os
import json
import random

import pandas as pd
import duckdb
import time
from tqdm import tqdm
import numpy as np
import argparse
import logging
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

# MOOCCube
# course.json --- id, core_id
# user.json

tqdm.pandas()


class AbstractMOOCDataPreprocessor(object):
    def __init__(self, args, data_dir, data_dir_processed, sep='\t'):
        self.args = args
        self.data_dir = data_dir
        self.data_dir_processed = data_dir_processed
        self.sep = sep
        self._init_path_()
        if not os.path.exists(self.data_dir_processed):
            os.makedirs(self.data_dir_processed)

    def _init_path_(self):
        self.relation_path = os.path.join(self.data_dir, 'relations')
        self.entities_path = os.path.join(self.data_dir, 'entities')

    def run_duckdb_sql(self, query):
        return duckdb.sql(query).df()

    def extract_course(self):
        raise NotImplemented

    def extract_kg_with_course(self, course_set):
        raise NotImplemented

    def course_per_user(self):
        raise NotImplemented

    def video_per_user(self):
        raise NotImplemented

    def video_per_course(self):
        raise NotImplemented

    def get_final(self, groups, course_index_set):
        with open(self.user_course_final_path_processed, 'w') as writer:
            for key, group in groups:
                user_id, = key
                user_index = str(user_id)
                courses = list(group['course_index'])

                cnt = 0
                for course_id in courses:
                    cnt += 1
                    course_index = str(course_id)
                    writer.write(user_index + '\t' + course_index + '\t' + '1' + '\n')

                unwatched_set = course_index_set - set(courses)
                replace = len(unwatched_set) < len(courses)
                cnt = 0
                for unwatched_course_id in np.random.choice(list(unwatched_set), size=len(courses), replace=replace):
                    cnt += 1
                    unwatched_course_index = str(unwatched_course_id)
                    writer.write(user_index + '\t' + unwatched_course_index + '\t' + '0' + '\n')

class MOOCCubeDataPreprocessor(AbstractMOOCDataPreprocessor):
    '''
    将不同course_id, 而core_id相同的课程视为同一门课程。将course_id都设置为core_id。
    '''
    def __init__(self, args, data_dir, data_dir_processed, origin_sep=',', sep='\t'):
        super(MOOCCubeDataPreprocessor, self).__init__(args, data_dir, data_dir_processed, sep)
        print('MOOCCubeDataPreprocessor...')
        self.init_path()

    def init_path(self):
        self.user_course_path = os.path.join(self.entities_path, 'user.json')
        self.user_course_path_processed = os.path.join(self.data_dir_processed, 'user-course.csv')
        self.user_course_filtered_path_processed = os.path.join(self.data_dir_processed, 'user-course_filtered.csv')
        self.user_course_filtered_final_path_processed = os.path.join(self.data_dir_processed, 'user-course_filtered_final')
        self.user_course_final_path_processed = os.path.join(self.data_dir_processed, 'user-course_final')

        self.user_video_path = os.path.join(self.relation_path, 'user-video.json')
        self.user_video_path_processed = os.path.join(self.data_dir_processed, 'user-video.csv')

        self.course_concept_path = os.path.join(self.relation_path, 'course-concept.json')
        self.course_concept_path_processed = os.path.join(self.data_dir_processed, 'course-concept.csv')

        self.course_video_path = os.path.join(self.relation_path, 'course-video.json')
        self.course_video_path_processed = os.path.join(self.data_dir_processed, 'course-video.csv')

        self.school_course_path = os.path.join(self.relation_path, 'school-course.json')
        self.school_course_path_processed = os.path.join(self.data_dir_processed, 'school-course.csv')

        self.teacher_course_path = os.path.join(self.relation_path, 'teacher-course.json')
        self.teacher_course_path_processed = os.path.join(self.data_dir_processed, 'teacher-course.csv')

        self.school_teacher_path = os.path.join(self.relation_path, 'school-teacher.json')
        self.school_teacher_path_processed = os.path.join(self.data_dir_processed, 'school-teacher.csv')

        self.video_concept_path = os.path.join(self.relation_path, 'video-concept.json')
        self.video_concept_path_processed = os.path.join(self.data_dir_processed, 'video-concept.csv')

        self.user_path_processed = os.path.join(self.data_dir_processed, 'user.csv')
        self.course_path_processed = os.path.join(self.data_dir_processed, 'course.csv')
        self.concept_path_processed = os.path.join(self.data_dir_processed, 'concept.csv')
        self.video_path_processed = os.path.join(self.data_dir_processed, 'video.csv')
        self.school_path_processed = os.path.join(self.data_dir_processed, 'school.csv')
        self.teacher_path_processed = os.path.join(self.data_dir_processed, 'teacher.csv')

        self.kg_path_processed = os.path.join(self.data_dir_processed, 'kg_final')
        self.entity_path_processed = os.path.join(self.data_dir_processed, 'entity.csv')
        self.relation_path_processed = os.path.join(self.data_dir_processed, 'relation.csv')


    def process(self):
        user_map, entity_map, course_set, video_set = self.extract_course()

        self.extract_kg_with_course_2(course_set=course_set, video=video_set, entity_map=entity_map)

    def extract_course(self):
        df_video_course = pd.read_csv(self.course_video_path, names=['course_id', 'video_id'], sep='\t')[['video_id', 'course_id']]
        df_user_videos = pd.read_csv(self.user_video_path, names=['user_id', 'video_id'], sep='\t')

        video_course_map = dict()
        for video_id, course_id in zip(df_video_course['video_id'], df_video_course['course_id']):
            video_course_map[video_id] = course_id

        def find_course(video):
            if video in video_course_map:
                return video_course_map[video]
            else:
                return None
        tmp_courses = df_user_videos['video_id'].progress_apply(lambda v: find_course(v))
        tmp_courses.rename('course_id', inplace=True)
        tmp_courses = tmp_courses[tmp_courses.notnull()]
        df_user_videos[tmp_courses.name] = tmp_courses

        df_user_course = pd.read_json(self.user_course_path, lines=True)[['id', 'course_order', 'enroll_time']].explode(['course_order', 'enroll_time'])
        df_user_course.columns = ['user_id', 'course_id', 'enroll_time']
        tmp_timestamp = df_user_course['enroll_time'].progress_apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
        tmp_timestamp.rename('timestamp', inplace=True)
        df_user_course[tmp_timestamp.name] = tmp_timestamp
        df_user_course.sort_values(['user_id', 'timestamp'], ascending=[True, True], inplace=True)

        sql_filter_n_core = f"""
            select uc.user_id, count(uc.course_id) ccnt
            from (select distinct user_id, course_id from df_user_course) uc
            inner join (select distinct user_id, course_id, count(video_id) vcnt from df_user_videos group by user_id, course_id having vcnt >= {self.args.n_core}) uv
            on uc.user_id=uv.user_id and uc.course_id=uv.course_id
            group by uc.user_id
            having ccnt >= {self.args.n_core}
            order by uc.user_id
        """
        n_core_user = self.run_duckdb_sql(sql_filter_n_core)[['user_id']]
        n_core_user_set = set(n_core_user['user_id'])

        sql_user_course = f"""
            select uc.user_id, uc.course_id, enroll_time, timestamp
            from (select distinct user_id, course_id, enroll_time, timestamp from df_user_course) uc
            inner join (select distinct user_id, course_id, count(video_id) vcnt from df_user_videos group by user_id, course_id having vcnt >= {self.args.n_core}) uv
            on uc.user_id=uv.user_id and uc.course_id=uv.course_id
            order by uc.user_id, timestamp
        """
        user_courses = self.run_duckdb_sql(sql_user_course)
        user_courses = user_courses[user_courses['user_id'].isin(n_core_user_set)]
        user_courses.to_csv(os.path.join(self.data_dir_processed, 'interaction-sorted.csv'), sep='\t', index=False)
        course_set = set(user_courses['course_id'])

        sql_user_videos_course = """
            select distinct uv.user_id, uv.video_id, uv.course_id
            from user_courses uc inner join df_user_videos uv on uc.user_id=uv.user_id and uc.course_id=uv.course_id  
        """
        user_videos_course = self.run_duckdb_sql(sql_user_videos_course)
        # user_videos_course = user_videos_course[user_videos_course['user_id'].isin(n_core_user_set) & user_videos_course['course_id'].isin(course_set)]
        video_set = set(user_videos_course['video_id'])
        course_video = user_videos_course[['course_id', 'video_id']].drop_duplicates()

        user_map = dict()
        for user_index, user_id in enumerate(n_core_user_set, start=0):
            user_map[user_id] = user_index

        entity_map = dict()
        entity_start_index = len(entity_map)
        for entity_index, entity_id in enumerate(course_set, start=entity_start_index):
            entity_map[entity_id] = entity_index

        entity_start_index = len(entity_map)
        for entity_index, entity_id in enumerate(video_set, start=entity_start_index):
            entity_map[entity_id] = entity_index

        tmp_user_index_list = n_core_user['user_id'].progress_apply(lambda x: user_map[x])
        tmp_user_index_list.rename('user_index', inplace=True)
        n_core_user[tmp_user_index_list.name] = tmp_user_index_list
        n_core_user.to_csv(self.user_path_processed, sep='\t', index=False)

        user_courses.loc[:, 'user_id'] = user_courses['user_id'].progress_apply(lambda x: user_map[x])
        user_courses.loc[:, 'course_id'] = user_courses['course_id'].progress_apply(lambda x: entity_map[x])
        user_courses.rename(columns={'user_id':'user_index', 'course_id':'course_index'}, inplace=True)
        user_courses.sort_values(['user_index', 'timestamp'], ascending=[True, True], inplace=True)
        user_courses.to_csv(self.user_course_filtered_path_processed, sep='\t', index=False,
                            header=False)

        user_courses['acted'] = 1
        user_courses.to_csv(self.user_course_filtered_final_path_processed, sep='\t', index=False,
                            columns=['user_index', 'course_index', 'acted'], header=False)
        course_index_set = set(user_courses['course_index'])
        groups = user_courses.groupby(['user_index'])
        self.get_final(groups=groups, course_index_set=course_index_set)

        user_videos_course.loc[:, 'user_id'] = user_videos_course['user_id'].progress_apply(lambda x: user_map[x])
        user_videos_course.loc[:, 'video_id'] = user_videos_course['video_id'].progress_apply(lambda x: entity_map[x])
        user_videos_course.loc[:, 'course_id'] = user_videos_course['course_id'].progress_apply(lambda x: entity_map[x])
        user_videos_course.rename(columns={'user_id':'user_index', 'video_id':'video_index', 'course_id':'course_index'}, inplace=True)
        user_videos_course.to_csv(self.user_video_path_processed, sep='\t', index=False)

        course_video.to_csv(self.course_video_path_processed, sep='\t', index=False)

        return user_map, entity_map, course_set, video_set

    def extract_kg_with_course_2(self, course_set, video, entity_map):
        relation_file_list = [
            'course-concept.json', 'course-video.json', 'school-course.json',
            'teacher-course.json', 'school-teacher.json', 'video-concept.json'
        ]
        concept = self.extract_kg_course_concept(course_set)
        # video = self.extract_kg_course_video(course_set)
        school = self.extract_kg_school_course(course_set)
        teacher = self.extract_kg_teacher_course(course_set)
        self.extract_kg_school_teacher(school, teacher)
        self.extract_kg_video_concept(video, concept)

        entity_set_list = [concept, school, teacher]
        entity_map = self.dense_index_2(entity_set_list, entity_map)

        relations = [
            ('course-concept', self.course_concept_path_processed),
            ('course-video', self.course_video_path_processed),
            ('school-course', self.school_course_path_processed),
            ('teacher-course', self.teacher_course_path_processed),
            ('school-teacher', self.school_teacher_path_processed),
            ('video-concept', self.video_concept_path_processed)
        ]
        self.construct_kg(relations, entity_map)
        return entity_map

    # 使用了课程映射
    def extract_kg_course_concept(self, course_set):
        concept = set()
        with open(self.course_concept_path_processed, 'w') as writer:
            writer.write('course_id' + '\t' + 'concept_id' + '\n')
            with open(self.course_concept_path, 'r') as reader:
                lines = reader.readlines()
                for line in tqdm(lines, desc='extract_kg_course_concept', ncols=80):
                    line = line.strip().split('\t')
                    course_id, concept_id = line[0], line[1]
                    # course_id = self.map2core(course_id)
                    if course_id not in course_set:
                        continue
                    concept.add(concept_id)
                    writer.write(course_id + '\t' + concept_id + '\n')
        with open(self.concept_path_processed, 'w') as writer:
            for concept_id in concept:
                writer.write(concept_id + '\n')
        return concept

    # 使用了课程映射
    def extract_kg_school_course(self, course_set):
        school = set()
        with open(self.school_course_path_processed, 'w') as writer:
            writer.write('school_id' + '\t' + 'course_id' + '\n')
            with open(self.school_course_path, 'r') as reader:
                lines = reader.readlines()
                for line in tqdm(lines, desc='extract_kg_school_course', ncols=80):
                    line = line.strip().split('\t')
                    school_id, course_id = line[0], line[1]
                    # course_id = self.map2core(course_id)
                    if course_id not in course_set:
                        continue
                    school.add(school_id)
                    writer.write(school_id + '\t' + course_id + '\n')

        with open(self.school_path_processed, 'w') as writer:
            for school_id in school:
                writer.write(school_id + '\n')
        return school

    # 使用了课程映射
    def extract_kg_teacher_course(self, course_set):
        teacher = set()
        with open(self.teacher_course_path_processed, 'w') as writer:
            writer.write('teacher_id' + '\t' + 'course_id' + '\n')
            with open(self.teacher_course_path, 'r') as reader:
                lines = reader.readlines()
                for line in tqdm(lines, desc='extract_kg_teacher_course', ncols=80):
                    line = line.strip().split('\t')
                    teacher_id, course_id = line[0], line[1]
                    # course_id = self.map2core(course_id)
                    if course_id not in course_set:
                        continue

                    teacher.add(teacher_id)
                    writer.write(teacher_id + '\t' + course_id + '\n')

        with open(self.teacher_path_processed, 'w') as writer:
            for teacher_id in teacher:
                writer.write(teacher_id + '\n')
        return teacher

    def extract_kg_school_teacher(self, school, teacher):
        with open(self.school_teacher_path_processed, 'w') as writer:
            writer.write('school_id' + '\t' + 'teacher_id' + '\n')
            with open(self.school_teacher_path, 'r') as reader:
                lines = reader.readlines()
                for line in tqdm(lines, desc='extract_kg_school_teacher', ncols=80):
                    line = line.strip().split('\t')
                    school_id, teacher_id = line[0], line[1]
                    if school_id not in school or teacher_id not in teacher:
                        continue
                    writer.write(school_id + '\t' + teacher_id + '\n')

    def extract_kg_video_concept(self, video, concept):
        with open(self.video_concept_path_processed, 'w') as writer:
            writer.write('video_id' + '\t' + 'concept_id' + '\n')
            with open(self.video_concept_path, 'r') as reader:
                lines = reader.readlines()
                for line in tqdm(lines, desc='extract_kg_video_concept', ncols=80):
                    line = line.strip().split('\t')
                    video_id, concept_id = line[0], line[1]
                    if video_id not in video or concept_id not in concept:
                        continue
                    writer.write(video_id + '\t' + concept_id + '\n')

    # 简单地使用了反向关系
    def construct_kg(self, relations, entity_map):
        relation_entity_map = dict()
        next_relation_index = 0
        with open(self.kg_path_processed, 'w') as writer:
            for relation, path in relations:
                if relation not in relation_entity_map:
                    relation_entity_map[relation] = next_relation_index
                    next_relation_index += 1
                relation_reverse = relation + '_reverse'
                if relation_reverse not in relation_entity_map:
                    relation_entity_map[relation_reverse] = next_relation_index
                    next_relation_index += 1
                relation = str(relation_entity_map[relation])
                relation_reverse = str(relation_entity_map[relation_reverse])
                with open(path, 'r') as reader:
                    lines = reader.readlines()
                    for line in lines[1:]:
                        line = line.strip().split('\t')
                        head, tail = str(entity_map[line[0]]), str(entity_map[line[1]])
                        writer.write(head + '\t' + relation + '\t' + tail + '\n')
                        writer.write(tail + '\t' + relation_reverse + '\t' + head + '\n')

        with open(self.relation_path_processed, 'w') as writer:
            writer.write('relation' + '\t' + 'index' + '\n')
            for relation, index in relation_entity_map.items():
                writer.write(relation + '\t' + str(index) + '\n')

    def dense_index_2(self, entity_set_list, entity_map):
        for entity_set in entity_set_list:
            entity_start_index = len(entity_map)
            for entity_index, entity_id in enumerate(entity_set, start=entity_start_index):
                entity_map[entity_id] = entity_index

        df_entity = pd.DataFrame.from_dict(entity_map, orient='index', columns=['entity_index']).reset_index()
        df_entity.rename(columns={'index': 'entity_id'}, inplace=True)
        df_entity.to_csv(self.entity_path_processed, sep='\t', index=False)
        return entity_map

    def map2core(self, x):
        if x in self.core_map:
            return self.core_map[x]
        else:
            return x

class MOOCCubeXDataPreprocessor(AbstractMOOCDataPreprocessor):
    def __init__(self, args, data_dir, data_dir_processed, origin_sep=',', sep='\t'):
        super(MOOCCubeXDataPreprocessor, self).__init__(args, data_dir, data_dir_processed, sep=sep)
        print('MOOCCubeXDataPreprocessor...')
        self.init_path()
        self.ccid_map = self.special_preprocess()

    def init_path(self):
        self.user_course_path = os.path.join(self.entities_path, 'user.json')
        self.user_course_path_processed = os.path.join(self.data_dir_processed, 'user-course.csv')
        self.user_course_filtered_path_processed = os.path.join(self.data_dir_processed, 'user-course_filtered.csv')
        self.user_course_filtered_final_path_processed = os.path.join(self.data_dir_processed, 'user-course_filtered_final')
        self.user_course_final_path_processed = os.path.join(self.data_dir_processed, 'user-course_final')

        self.user_video_path = os.path.join(self.relation_path, 'user-video.json')
        self.user_video_path_processed = os.path.join(self.data_dir_processed, 'user-video.csv')
        #
        # self.course_concept_path = os.path.join(self.relation_path, 'course-concept.json')
        self.course_concept_path_processed = os.path.join(self.data_dir_processed, 'course-concept.csv')
        #
        self.course_video_path = os.path.join(self.entities_path, 'course.json')
        self.course_video_path_processed = os.path.join(self.data_dir_processed, 'course-video.csv')
        #
        self.school_course_path = os.path.join(self.relation_path, 'course-school.txt')
        self.school_course_path_processed = os.path.join(self.data_dir_processed, 'school-course.csv')
        #
        self.teacher_course_path = os.path.join(self.relation_path, 'course-teacher.txt')
        self.teacher_course_path_processed = os.path.join(self.data_dir_processed, 'teacher-course.csv')
        #
        self.school_path = os.path.join(self.entities_path, 'school.json')
        self.teacher_path = os.path.join(self.entities_path, 'teacher.json')
        self.school_teacher_path_processed = os.path.join(self.data_dir_processed, 'school-teacher.csv')
        #
        self.video_concept_path = os.path.join(self.relation_path, 'concept-video.txt')
        self.video_concept_path_processed = os.path.join(self.data_dir_processed, 'video-concept.csv')
        #
        self.course_field_path = os.path.join(self.relation_path, 'course-field.json')
        self.course_field_path_processed = os.path.join(self.data_dir_processed, 'course-field.csv')
        #
        self.user_path_processed = os.path.join(self.data_dir_processed, 'user.csv')
        self.course_path_processed = os.path.join(self.data_dir_processed, 'course.csv')
        self.concept_path_processed = os.path.join(self.data_dir_processed, 'concept.csv')
        self.video_path_processed = os.path.join(self.data_dir_processed, 'video.csv')
        self.school_path_processed = os.path.join(self.data_dir_processed, 'school.csv')
        self.teacher_path_processed = os.path.join(self.data_dir_processed, 'teacher.csv')
        #
        self.field_path_processed = os.path.join(self.data_dir_processed, 'field.csv')
        #
        self.kg_path_processed = os.path.join(self.data_dir_processed, 'kg_final')
        self.entity_path_processed = os.path.join(self.data_dir_processed, 'entity.csv')
        self.relation_path_processed = os.path.join(self.data_dir_processed, 'relation.csv')

    def special_preprocess(self):
        video_ccid_df = pd.read_csv(os.path.join(self.relation_path, 'video_id-ccid.txt'), sep='\t', names=['video_id', 'ccid'])
        video_ccid_df = video_ccid_df.to_dict('list')
        ccid_map = {}
        for video_id, ccid in zip(video_ccid_df['video_id'], video_ccid_df['ccid']):
            if video_id not in ccid_map:
                ccid_map[video_id] = ccid

        return ccid_map

    def process(self):
        user_map, entity_map, course_set, video_set = self.extract_course()

        self.extract_kg_with_course_2(course_set, video_set, entity_map)

    def extract_course(self):
        df_user_course = pd.read_json(self.user_course_path, lines=True)[['id', 'course_order', 'enroll_time']]
        df_user_course = df_user_course.explode(['course_order', 'enroll_time']).reset_index(drop=True)
        df_user_course.columns = ['user_id', 'course_id', 'enroll_time']
        tmp_course = df_user_course['course_id'].progress_apply(lambda x: 'C_' + str(x))
        df_user_course.loc[:, 'course_id'] = tmp_course
        tmp_timestamp = df_user_course['enroll_time'].progress_apply(lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
        tmp_timestamp.rename('timestamp', inplace=True)
        df_user_course[tmp_timestamp.name] = tmp_timestamp
        df_user_course.sort_values(['user_id', 'timestamp'], ascending=[True, True], inplace=True)

        def video_id_map_to_ccid(video_id):
            if video_id in self.ccid_map:
                return self.ccid_map[video_id]
            else:
                return None
        df_video_course = pd.read_json(self.course_video_path, lines=True)[['id', 'resource']].explode(['resource']).reset_index(drop=True)

        resource_tmp = df_video_course['resource'].progress_apply(lambda x: x['resource_id'])
        df_video_course.loc[:, 'resource'] = resource_tmp
        df_video_course = df_video_course[df_video_course['resource'].str.startswith('V_')].reset_index(drop=True)[['resource', 'id']]
        df_video_course.columns = ['video_id', 'course_id']
        tmp_video = df_video_course['video_id'].progress_apply(lambda x: video_id_map_to_ccid(x))
        df_video_course.loc[:, 'video_id'] = tmp_video
        df_video_course = df_video_course.drop_duplicates()
        df_video_course = df_video_course[df_video_course['video_id'].notnull()]

        video_map = dict()
        for video_id, course_id in zip(df_video_course['video_id'], df_video_course['course_id']):
            video_map[video_id] = course_id

        def find_course_id(x):
            if x in video_map:
                return video_map[x]
            else:
                return None

        df_user_videos = pd.read_json(self.user_video_path, lines=True)[['user_id', 'seq']].explode(['seq']).reset_index(drop=True)
        video_tmp = df_user_videos['seq'].progress_apply(lambda x: x['video_id'])
        video_tmp = video_tmp.progress_apply(lambda x: video_id_map_to_ccid(x))
        df_user_videos.loc[:, 'seq'] = video_tmp
        df_user_videos.columns = ['user_id', 'video_id']

        course_tmp = df_user_videos['video_id'].progress_apply(lambda x: find_course_id(x))
        df_user_videos['course_id'] = course_tmp
        df_user_videos = df_user_videos[df_user_videos['course_id'].notnull()].reset_index(drop=True)


        sql_filter_n_core = f"""
        select user_id, count(course_id) ccnt
        from(
        select distinct user_id, course_id, count(video_id) vcnt
        from df_user_videos
        group by user_id, course_id
        having vcnt >= {self.args.n_core}
        ) group by user_id
        having ccnt >= {self.args.n_core}
        order by user_id
        """
        # sql_filter_n_core = f"""
        # select uc.user_id, count(uc.course_id) ccnt
        # from (select distinct user_id, course_id from df_user_course) uc
        # inner join (select distinct user_id, course_id, count(video_id) vcnt from df_user_videos group by user_id, course_id having vcnt >= {self.args.n_core}) uv
        # on uc.user_id=uv.user_id and uc.course_id=uv.course_id
        # group by uc.user_id
        # having ccnt >= {self.args.n_core}
        # order by uc.user_id
        # """
        n_core_user = self.run_duckdb_sql(sql_filter_n_core)[['user_id']]
        n_core_user_set = set(n_core_user['user_id'])

        sql_user_course = f"""
        select user_id, course_id
        from(
        select distinct user_id, course_id, count(video_id) vcnt
        from df_user_videos
        group by user_id, course_id
        having vcnt >= {self.args.n_core}
        )
        order by user_id
        """
        # sql_user_course = f"""
        # select uc.user_id, uc.course_id, enroll_time, timestamp
        # from (select distinct user_id, course_id, enroll_time, timestamp from df_user_course) uc
        # inner join (select distinct user_id, course_id, count(video_id) vcnt from df_user_videos group by user_id, course_id having vcnt >= {self.args.n_core}) uv
        # on uc.user_id=uv.user_id and uc.course_id=uv.course_id
        # order by uc.user_id, timestamp
        # """
        user_courses = self.run_duckdb_sql(sql_user_course)
        user_courses = user_courses[user_courses['user_id'].isin(n_core_user_set)]
        user_courses.to_csv(os.path.join(self.data_dir_processed, 'interaction-sorted.csv'), sep='\t', index=False)
        course_set = set(user_courses['course_id'])

        sql_user_videos_course = """
        select distinct uv.user_id, uv.video_id, uv.course_id
        from user_courses uc inner join df_user_videos uv on uc.user_id=uv.user_id and uc.course_id=uv.course_id
        """
        user_videos_course = self.run_duckdb_sql(sql_user_videos_course)
        video_set = set(user_videos_course['video_id'])
        course_video = user_videos_course[['course_id', 'video_id']].drop_duplicates()

        user_map = dict()
        for user_index, user_id in enumerate(n_core_user_set, start=0):
            user_map[user_id] = user_index

        entity_map = dict()
        entity_start_index = len(entity_map)
        for entity_index, entity_id in enumerate(course_set, start=entity_start_index):
            entity_map[entity_id] = entity_index

        entity_start_index = len(entity_map)
        for entity_index, entity_id in enumerate(video_set, start=entity_start_index):
            entity_map[entity_id] = entity_index

        tmp_user_index_list = n_core_user['user_id'].progress_apply(lambda x: user_map[x])
        tmp_user_index_list.rename('user_index', inplace=True)
        n_core_user[tmp_user_index_list.name] = tmp_user_index_list
        n_core_user.to_csv(self.user_path_processed, sep='\t', index=False)

        user_courses.loc[:, 'user_id'] = user_courses['user_id'].progress_apply(lambda x: user_map[x])
        user_courses.loc[:, 'course_id'] = user_courses['course_id'].progress_apply(lambda x: entity_map[x])
        user_courses.rename(columns={'user_id': 'user_index', 'course_id': 'course_index'}, inplace=True)
        # user_courses.sort_values(['user_index', 'timestamp'], ascending=[True, True], inplace=True)
        user_courses.sort_values(['user_index'], ascending=[True], inplace=True)
        user_courses.to_csv(self.user_course_filtered_path_processed, sep='\t', index=False,
                            header=False)

        user_courses['acted'] = 1
        user_courses.to_csv(self.user_course_filtered_final_path_processed, sep='\t', index=False,
                            columns=['user_index', 'course_index', 'acted'], header=False)
        course_index_set = set(user_courses['course_index'])
        groups = user_courses.groupby(['user_index'])
        self.get_final(groups=groups, course_index_set=course_index_set)


        user_videos_course.loc[:, 'user_id'] = user_videos_course['user_id'].progress_apply(lambda x: user_map[x])
        user_videos_course.loc[:, 'video_id'] = user_videos_course['video_id'].progress_apply(lambda x: entity_map[x])
        user_videos_course.loc[:, 'course_id'] = user_videos_course['course_id'].progress_apply(lambda x: entity_map[x])
        user_videos_course.rename(
            columns={'user_id': 'user_index', 'video_id': 'video_index', 'course_id': 'course_index'}, inplace=True)
        user_videos_course.to_csv(self.user_video_path_processed, sep='\t', index=False)

        course_video.to_csv(self.course_video_path_processed, sep='\t', index=False)

        return user_map, entity_map, course_set, video_set


    def extract_kg_with_course_2(self, course_set, video, entity_map):
        relation_file_list = [
            'concept-video.txt', 'course-field.json', 'course-school.txt',
            'course-teacher.txt', 'user-video.json', 'concept-video.txt'
        ]
        # video = self.extract_kg_course_video(course_set)
        school = self.extract_kg_school_course(course_set)
        teacher = self.extract_kg_teacher_course(course_set)
        self.extract_kg_school_teacher(school, teacher)
        concept = self.extract_kg_video_concept(video)
        field = self.extract_kg_course_field(course_set)
        self.extract_kg_course_concept()

        entity_set_list = [concept, school, teacher, field]
        entity_map = self.dense_index_2(entity_set_list, entity_map)

        relations = [
            ('course_video', self.course_video_path_processed),
            ('course_concept', self.course_concept_path_processed),
            ('school-course', self.school_course_path_processed),
            ('teacher-course', self.teacher_course_path_processed),
            ('school-teacher', self.school_teacher_path_processed),
            ('video-concept', self.video_concept_path_processed),
            ('course-field', self.course_field_path_processed)
        ]
        self.construct_kg(relations, entity_map)
        return entity_map

    def extract_kg_school_course(self, course_set):
        logging.info('extract_kg_school_course...')
        school_course_df = pd.read_csv(self.school_course_path, names=['course_id', 'school_id'], sep='\t')[['school_id', 'course_id']]
        tmp_school_course = school_course_df[school_course_df['course_id'].isin(course_set)]
        school_course_df = tmp_school_course
        school_course_df.to_csv(self.school_course_path_processed,
                             index=False,
                             sep='\t')
        school = list(set(school_course_df['school_id']))
        school_df = pd.DataFrame({'school_id': school})
        school_df.to_csv(self.school_path_processed,
                         index=False,
                         sep='\t')

        return school

    def extract_kg_teacher_course(self, course_set):
        logging.info('extract_kg_teacher_course...')
        teacher_course_df = pd.read_csv(self.teacher_course_path, names=['course_id', 'teacher_id'], sep='\t')[['teacher_id', 'course_id']]
        tmp_teacher_course = teacher_course_df[teacher_course_df['course_id'].isin(course_set)]
        teacher_course_df = tmp_teacher_course
        teacher_course_df.to_csv(self.teacher_course_path_processed,
                                 index=False,
                                 sep='\t')

        teacher = list(set(teacher_course_df['teacher_id']))
        teacher_df = pd.DataFrame({'teacher_id': teacher})
        teacher_df.to_csv(self.teacher_path_processed,
                          index=False,
                          sep='\t')
        return teacher

    def extract_kg_school_teacher(self, school, teacher):
        logging.info('extract_kg_school_teacher...')
        school_df = pd.read_json(self.school_path, lines=True)[['id', 'name']]
        school_df.columns = ['school_id', 'school_name']

        teacher_df = pd.read_json(self.teacher_path, lines=True)[['id', 'org_name']]
        teacher_df.columns = ['teacher_id', 'school_name']

        school_teacher_df = pd.merge(school_df, teacher_df, on='school_name')
        tmp_school_teacher = school_teacher_df[school_teacher_df['school_id'].isin(school) & school_teacher_df['teacher_id'].isin(teacher)][['school_id', 'teacher_id']]
        school_teacher_df = tmp_school_teacher
        school_teacher_df.to_csv(self.school_teacher_path_processed,
                                 index=False,
                                 sep='\t')

    def extract_kg_video_concept(self, video_set):
        logging.info('extract_kg_video_concept...')
        video_concept_df = pd.read_csv(self.video_concept_path, names=['concept_id', 'video_id'], sep='\t')[['video_id', 'concept_id']]
        tmp_video_concept = video_concept_df[video_concept_df['video_id'].isin(video_set)]
        video_concept_df = tmp_video_concept
        video_concept_df.to_csv(self.video_concept_path_processed,
                             index=False,
                             sep='\t')

        concept = list(set(video_concept_df['concept_id']))
        concept_df = pd.DataFrame({'concept_id': concept})
        concept_df.to_csv(self.concept_path_processed,
                          index=False,
                          sep='\t')

        return concept

    def extract_kg_course_field(self, course_set):
        logging.info('extract_kg_course_field...')
        course_field_df = pd.read_json(self.course_field_path, lines=True).explode(['field'])[['course_id', 'field']]
        course_field_df.columns = ['course_id', 'field_id']
        tmp_course_id = course_field_df['course_id'].progress_apply(lambda x: 'C_'+str(x))
        course_field_df.loc[:, 'course_id'] = tmp_course_id

        tmp_course_field = course_field_df[course_field_df['course_id'].isin(course_set)]
        course_field_df = tmp_course_field
        course_field_df.to_csv(self.course_field_path_processed,
                               index=False,
                               sep='\t')

        field = list(set(course_field_df['field_id']))
        field_df = pd.DataFrame({'field_id': field})
        field_df.to_csv(self.field_path_processed,
                        index=False,
                        sep='\t')

        return field

    def extract_kg_course_concept(self):
        cv_df = pd.read_csv(self.course_video_path_processed, sep='\t')
        vc_df = pd.read_csv(self.video_concept_path_processed, sep='\t')

        course_videos = dict()
        for course_id, video_id in zip(cv_df['course_id'], cv_df['video_id']):
            if course_id not in course_videos:
                course_videos[course_id] = set()
            course_videos[course_id].add(video_id)

        video_concept = dict()
        for video_id, concept_id in zip(vc_df['video_id'], vc_df['concept_id']):
            if video_id not in video_concept:
                video_concept[video_id] = set()
            video_concept[video_id].add(concept_id)

        course_concepts = dict()
        for course_id, video_ids in course_videos.items():
            for video_id in video_ids:
                if video_id in video_concept:
                    if course_id not in course_concepts:
                        course_concepts[course_id] = set()
                    course_concepts[course_id] |= video_concept[video_id]

        with open(self.course_concept_path_processed, 'w') as writer:
            writer.write(f"course_id\tconcept_id\n")
            for course_id, concept_ids in course_concepts.items():
                for concept_id in concept_ids:
                    writer.write(f"{course_id}\t{concept_id}\n")

    def construct_kg(self, relations, entity_map):
        logging.info('construct_kg...')
        relation_entity_map = dict()
        next_relation_index = 0
        head_list = []
        relation_list = []
        tail_list = []
        for relation, path in relations:
            if relation not in relation_entity_map:
                relation_entity_map[relation] = next_relation_index
                next_relation_index += 1
            relation_reverse = relation + '_reverse'
            if relation_reverse not in relation_entity_map:
                relation_entity_map[relation_reverse] = next_relation_index
                next_relation_index += 1
            df = pd.read_csv(path, sep='\t')
            df.columns = ['head', 'tail']
            df_dict = df.to_dict('list')
            for head, tail in zip(df_dict['head'], df_dict['tail']):
                head = entity_map[head]
                tail = entity_map[tail]

                head_list.append(head)
                relation_list.append(relation_entity_map[relation])
                tail_list.append(tail)

                head_list.append(tail)
                relation_list.append(relation_entity_map[relation_reverse])
                tail_list.append(head)
        kg_df = pd.DataFrame({'head': head_list, 'relation': relation_list, 'tail': tail_list})
        kg_df.to_csv(self.kg_path_processed,
                  header=None,
                  index=False,
                  sep='\t')

        with open(self.relation_path_processed, 'w') as writer:
            writer.write('relation' + '\t' + 'index' + '\n')
            for relation, index in relation_entity_map.items():
                writer.write(relation + '\t' + str(index) + '\n')

    def dense_index_2(self, entity_set_list, entity_map):
        for entity_set in entity_set_list:
            entity_start_index = len(entity_map)
            for entity_index, entity_id in enumerate(entity_set, start=entity_start_index):
                entity_map[entity_id] = entity_index

        df_entity = pd.DataFrame.from_dict(entity_map, orient='index', columns=['entity_index']).reset_index()
        df_entity.rename(columns={'index': 'entity_id'}, inplace=True)
        df_entity.to_csv(self.entity_path_processed, sep='\t', index=False)
        return entity_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MOOCCubeX', help='which dataset to preprocess')
    parser.add_argument('--data_dir_processed', type=str, default='data_processed', help='')
    parser.add_argument('--n_core', type=int, default=5, help='')
    args = parser.parse_args()
    print(args)

    np.random.seed(2024)
    random.seed(2024)

    if args.dataset == 'mooccube' or args.dataset == 'MOOCCube' :
        preprocesser = MOOCCubeDataPreprocessor(args, data_dir='data/MOOCCube', data_dir_processed=args.data_dir_processed + '/MOOCCube')
    elif args.dataset == 'mooccubex' or args.dataset == 'MOOCCubeX':
        preprocesser = MOOCCubeXDataPreprocessor(args, data_dir='data/MOOCCubeX',data_dir_processed=args.data_dir_processed + '/MOOCCubeX')
    preprocesser.process()

