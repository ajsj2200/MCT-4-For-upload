### UCL TimeSeries Utils _ 2021.09.06. ###

import os
import pandas as pd
import numpy as np

import multiprocessing
from tqdm import tqdm
import parmap
from ts_dist import edr_dist
from tslearn.metrics import dtw as DynamicTimeWarping

def toSMA(df, window='5s'):
    '''
    시계열의 단순이동평균을 구합니다.
    
    df : 시계열 데이터프레임
    window : 단순이동평균의 샘플링 크기
    '''
    dates = pd.date_range(start=df.index.min().ceil('s'), end=df.index.max().ceil('s'), freq='s')
    rolling_helper = pd.DataFrame(dates, index=dates, columns=['rolling_helper'])
    
    data = pd.merge(df, rolling_helper, left_index=True, right_index=True, how='outer').sort_index().drop(columns='rolling_helper')
    data = data.rolling(window).mean()
    data = data[data.index.microsecond == 0]
    return data

class ts_pattern:
    def __init__(self, data, data_search):
        '''
        data : 시계열 데이터프레임
        data_serch : 기준 패턴 데이터프레임
        '''
        
        self.data = data.sort_index()
        self.data_search = data_search.sort_index()
        self.similarity_table = None
        self.pattern_list = None
        
        self.num_cores = os.cpu_count()
        self.window = data_search.index.max().ceil('s') - data_search.index.min().floor('s')
        
    def fit(self, algorithm, search_dense = '1s', allowable_scale_rate = 0.7, epsilon=0.5):
        '''
        기준 패턴으로부터 시계열의 유사도 테이블을 생성합니다.

        algorithm : 유사도 측정 알고리즘 선택 (euclidean / edr / dtw)
        search_dense : 패턴 시작 지점 판별 간격
        allowable_scale_rate : 패턴이 기준 패턴과 대비하여 가져야 할 최소한의 스케일
        epsilon : edr 측정 시 민감도 (작을수록 엄격)
        '''
        dates = pd.date_range(start=self.data.index.min().floor('s'), end=self.data.index.max().ceil('s'), freq=search_dense)
        dates_map = list(self.daterange_split(dates, round(len(dates) / self.num_cores)))

        if algorithm == 'euclidean':
            result = parmap.map(self.calculate_euclidean, dates_map, allowable_scale_rate, pm_pbar=True, pm_processes=self.num_cores)
        elif algorithm == 'edr':
            result = parmap.map(self.calculate_edr, dates_map, epsilon, allowable_scale_rate, pm_pbar=True, pm_processes=self.num_cores)
        elif algorithm == 'dtw':
            result = parmap.map(self.calculate_dtw, dates_map, allowable_scale_rate, pm_pbar=True, pm_processes=self.num_cores)
        else:
            raise ValueError("Algorithm must be 'euclidean' or 'edr' or 'dtw'")

        index = []
        similarity_values = []
        for r in result:
            for i in range(len(r)):
                index.append(r[i][0])
                similarity_values.append(r[i][1])

        self.similarity_table = pd.DataFrame(similarity_values, index=index, columns=[algorithm])
        return self.similarity_table
    
    """
    def fit_edr(self, search_dense = '1s', allowable_scale_rate = 0.7, epsilon = 0.5):
        '''
        기준 패턴으로부터 시계열의 EDR 테이블을 생성합니다.

        search_dense : 패턴 시작 지점 판별 간격
        allowable_scale_rate : 패턴이 기준 패턴과 대비하여 가져야 할 최소한의 스케일
        epsilon : edr 측정 시 민감도 (작을수록 엄격)
        '''
        dates = pd.date_range(start=self.data.index.min().floor('s'), end=self.data.index.max().ceil('s'), freq=search_dense)
        dates_map = list(self.daterange_split(dates, round(len(dates) / self.num_cores)))
        
        result = parmap.map(self.calculate_edr, dates_map, epsilon, allowable_scale_rate, pm_pbar=True, pm_processes=self.num_cores)

        index = []
        edr_values = []
        for r in result:
            for i in range(len(r)):
                index.append(r[i][0])
                edr_values.append(r[i][1])

        self.similarity_table = pd.DataFrame(edr_values, index=index, columns=['edr'])
        return self.similarity_table

    def fit_dtw(self, search_dense = '1s', allowable_scale_rate = 0.7):
        '''
        기준 패턴으로부터 시계열의 DTW 테이블을 생성합니다.

        search_dense : 패턴 시작 지점 판별 간격
        allowable_scale_rate : 패턴이 기준 패턴과 대비하여 가져야 할 최소한의 스케일
        '''
        dates = pd.date_range(start=self.data.index.min().floor('s'), end=self.data.index.max().ceil('s'), freq=search_dense)
        dates_map = list(self.daterange_split(dates, round(len(dates) / self.num_cores)))

        result = parmap.map(self.calculate_dtw, dates_map, allowable_scale_rate, pm_pbar=True, pm_processes=self.num_cores)

        index = []
        dtw_values = []
        for r in result:
            for i in range(len(r)):
                index.append(r[i][0])
                dtw_values.append(r[i][1])

        self.similarity_table = pd.DataFrame(dtw_values, index=index, columns=['dtw'])
        return self.similarity_table
        """
    
    def search_pattern(self, threshold):
        '''
        생성한 유사도 테이블로부터 패턴을 찾습니다.
        
        threshold : 패턴의 임계 유사도 (작을수록 엄격, EDR: 0 ~ 1, DTW: 0 ~ inf.)
        '''
        
        pattern_similarity = []
        pattern_index = []
        column_name = self.similarity_table.columns[0]
        search_point = self.similarity_table.index.min().floor('s')

        for t in tqdm(self.similarity_table.index):
            if t < search_point:
                continue
            
            similarity_min = self.similarity_table.loc[t:t+self.window*0.6, column_name].min()
            index_min = self.similarity_table.loc[t:t+self.window*0.6, column_name].idxmin()
            
            if not index_min is np.nan:
                if self.similarity_table.loc[index_min:index_min+self.window*0.6, column_name].idxmin() != index_min:
                    search_point = index_min
                    continue

            pattern_similarity.append(similarity_min)
            pattern_index.append(index_min)
            
            search_point = index_min + self.window*0.6

        pattern_list = pd.DataFrame(pattern_similarity, index=pattern_index, columns=[column_name])
        self.pattern_list = pattern_list[pattern_list < threshold].dropna()
        
        return self.pattern_list
    
    def pattern_extract(self, data):
        '''
        찾은 패턴을 추출합니다.
        '''
        data = data.sort_index()
        sample = []
        
        for index in tqdm(self.pattern_list.index):
            if not data.loc[index:index+self.window].empty:
                sample.append(data.loc[index:index+self.window])
            
        return sample

    
    ######--- 보조함수 ---######
    def calculate_euclidean(self, daterange, allowable_scale_rate):
        find = []
        min_scale = self.data_search.mean() * allowable_scale_rate
        max_scale = self.data_search.mean() / allowable_scale_rate
        
        data_search = self.data_search.reset_index(drop=True)

        for t in daterange:
            data_cal = self.data.loc[t:t+self.window]

            if (data_cal.mean() >= min_scale).sum() != 0 and (data_cal.mean() <= max_scale).sum() != 0:
                euclidean = np.linalg.norm(data_search - data_cal.reset_index(drop=True))
                point = (t, euclidean)
                find.append(point)

        return find
    
    def calculate_edr(self, daterange, epsilon, allowable_scale_rate):
        find = []
        min_scale = self.data_search.mean() * allowable_scale_rate
        max_scale = self.data_search.mean() / allowable_scale_rate

        for t in daterange:
            data_cal = self.data.loc[t:t+self.window]

            if (data_cal.mean() >= min_scale).sum() != 0 and (data_cal.mean() <= max_scale).sum() != 0:
                edr = edr_dist(self.data_search.T, data_cal.T, epsilon = epsilon)
                point = (t, edr)
                find.append(point)

        return find

    def calculate_dtw(self, daterange, allowable_scale_rate):
        find = []
        min_scale = self.data_search.mean() * allowable_scale_rate
        max_scale = self.data_search.mean() / allowable_scale_rate

        for t in daterange:
            data_cal = self.data.loc[t:t+self.window]

            if (data_cal.mean() >= min_scale).sum() != 0 and (data_cal.mean() <= max_scale).sum() != 0:
                dtw = DynamicTimeWarping(self.data_search, data_cal)
                point = (t, dtw)
                find.append(point)

        return find

    def daterange_split(self, daterange, size):
        for r in range(0, len(daterange), size): 
            yield daterange[r : r+size]

class similarity_filter:
    def __init__(self, sample_list, base_pattern, algorithm, epsilon=0.5):
        '''
        sample_list : 시계열 데이터프레임 리스트
        base_pattern : 기준 패턴 데이터프레임
        algorithm : 유사도 측정 알고리즘 선택 (euclidean / edr / dtw)
        epsilon : edr 측정 시 민감도 (작을수록 엄격)
        '''
        
        self.sample_list = sample_list
        self.base_pattern = base_pattern
        self.pattern_list = None
        
        self.window = base_pattern.index.max().ceil('s') - base_pattern.index.min().floor('s')
        self.algorithm = algorithm
        
        sample_index = []
        sample_similarity = []
        
        if algorithm == 'euclidean':
            for sample in tqdm(sample_list):
                sample_index.append(sample.index.min())
                sample_similarity.append(np.linalg.norm(sample - base_pattern))
        elif algorithm == 'edr':
            for sample in tqdm(sample_list):
                sample_index.append(sample.index.min())
                sample_similarity.append(edr_dist(sample, base_pattern, epsilon=epsilon))
        elif algorithm == 'dtw':
            for sample in tqdm(sample_list):
                sample_index.append(sample.index.min())
                sample_similarity.append(DynamicTimeWarping(sample, base_pattern))
        else:
            raise ValueError("Algorithm must be 'euclidean' or 'edr' or 'dtw'")
        
        self.similarity_table = pd.DataFrame(sample_similarity, index=sample_index, columns=[algorithm])
    
    def fit(self, threshold):
        self.pattern_list = self.similarity_table.loc[self.similarity_table[self.algorithm] < threshold]
        return self.pattern_list
    
    def pattern_extract(self, data):
        '''
        찾은 패턴을 추출합니다.
        '''
        sample = []
        
        for index in tqdm(self.pattern_list.index):
            sample.append(data.loc[index:index+self.window])
            
        return sample


class live_pattern_detect:
    def __init__(self, base_pattern, input_num):
        self.base_pattern = base_pattern
        self.base_data = pd.DataFrame()
        self.data = [pd.DataFrame() for _ in range(input_num)]
        self.similarity_table = pd.DataFrame()
        self.pattern_list = pd.DataFrame()
        
        self.window = base_pattern.index.max().ceil('s') - base_pattern.index.min().floor('s')
        self.base_data_buff = pd.DataFrame()
        self.similarity_buff = None
        self.check_point = None
        self.search_point = pd.Timestamp(0)
        
        self.t = None # 임시 변수
    
    def set_nowTime(self, t): # 임시 함수
        self.t = t
    
    def put_data(self, base_data, data_list, base_sma='1s'):
        self.check_point = base_data.index.min().ceil('s')
        
        self.base_data_buff.drop(self.base_data_buff.loc[:self.check_point-pd.Timedelta(base_sma)].index, inplace=True)
        self.base_data_buff = base_data.combine_first(self.base_data_buff)
        
        sma_buff = toSMA(self.base_data_buff, window=base_sma)
        
        self.base_data.drop(self.base_data.loc[:self.t-self.window*7].index, inplace=True)
        self.base_data = sma_buff.combine_first(self.base_data).interpolate(method='time')
        
        for i in range(len(data_list)):
            self.data[i].drop(self.data[i].loc[:self.t-self.window*7].index, inplace=True)
            self.data[i] = data_list[i].combine_first(self.data[i])
    
    def calculate_euclidean(self, search_dense='1s', allowable_scale_rate=0.7):
        result = []
        daterange = pd.date_range(start=self.check_point-self.window, end=self.base_data.index.max()-self.window, freq=search_dense)
        min_scale = self.base_pattern.mean() * allowable_scale_rate
        max_scale = self.base_pattern.mean() / allowable_scale_rate
        
        base_pattern = self.base_pattern.reset_index(drop=True)

        for t in daterange:
            data_cal = self.base_data.loc[t:t+self.window]

            if (data_cal.mean() > min_scale).sum() != 0 and (data_cal.mean() < max_scale).sum() != 0:
                euclidean = np.linalg.norm(base_pattern - data_cal.reset_index(drop=True))
                point = (t, euclidean)
                result.append(point)

        index = []
        similarity_values = []
        for r in result:
            index.append(r[0])
            similarity_values.append(r[1])

        self.similarity_buff = pd.DataFrame(similarity_values, index=index, columns=['euclidean'])
        
        self.similarity_table.drop(self.similarity_table.loc[:self.t-self.window*7].index, inplace=True)
        self.similarity_table = self.similarity_buff.combine_first(self.similarity_table)
    
    def calculate_edr(self, search_dense='1s', allowable_scale_rate=0.7, epsilon=0.5):
        result = []
        daterange = pd.date_range(start=self.data.index.min().floor('s'), end=self.data.index.max().ceil('s'), freq=search_dense)
        min_scale = self.base_pattern.mean() * allowable_scale_rate
        max_scale = self.base_pattern.mean() / allowable_scale_rate

        for t in daterange:
            data_cal = self.base_data.loc[t:t+self.window]

            if (data_cal.mean() > min_scale).sum() != 0 and (data_cal.mean() < max_scale).sum() != 0:
                edr = edr_dist(self.base_pattern.T, data_cal.T, epsilon = epsilon)
                point = (t, edr)
                result.append(point)

        index = []
        similarity_values = []
        for r in result:
            index.append(r[0])
            similarity_values.append(r[1])
        
        self.similarity_buff = pd.DataFrame(similarity_values, index=index, columns=['edr'])
        
        self.similarity_table.drop(self.similarity_table.loc[:self.t-self.window*7].index, inplace=True)
        self.similarity_table = self.similarity_buff.combine_first(self.similarity_table)

    def calculate_dtw(self, search_dense='1s', allowable_scale_rate=0.7):
        result = []
        daterange = pd.date_range(start=self.data.index.min().floor('s'), end=self.data.index.max().ceil('s'), freq=search_dense)
        min_scale = self.base_pattern.mean() * allowable_scale_rate
        max_scale = self.base_pattern.mean() / allowable_scale_rate

        for t in daterange:
            data_cal = self.base_data.loc[t:t+self.window]

            if (data_cal.mean() > min_scale).sum() != 0 and (data_cal.mean() < max_scale).sum() != 0:
                dtw = DynamicTimeWarping(self.base_pattern, data_cal)
                point = (t, dtw)
                result.append(point)

        index = []
        similarity_values = []
        for r in result:
            index.append(r[0])
            similarity_values.append(r[1])

        self.similarity_buff = pd.DataFrame(similarity_values, index=index, columns=['dtw'])
        
        self.similarity_table.drop(self.similarity_table.loc[:self.t-self.window*7].index, inplace=True)
        self.similarity_table = self.similarity_buff.combine_first(self.similarity_table)
        
    def detect_pattern(self, threshold):
        pattern_similarity = []
        pattern_index = []
        column_name = self.similarity_table.columns[0]

        for i in self.similarity_buff.index:
            t = i - self.window*1.2
            if t < self.search_point:
                continue
            
            if self.similarity_table.loc[t:t+self.window*0.6, column_name].empty:
                continue
            
            similarity_min = self.similarity_table.loc[t:t+self.window*0.6, column_name].min()
            index_min = self.similarity_table.loc[t:t+self.window*0.6, column_name].idxmin()

            if similarity_min < threshold:
                if self.similarity_table.loc[index_min:index_min+self.window*0.6, column_name].idxmin() != index_min:
                    self.search_point = index_min
                    continue

                pattern_similarity.append(similarity_min)
                pattern_index.append(index_min)

                self.search_point = index_min + self.window*0.6

        pattern_list = pd.DataFrame(pattern_similarity, index=pattern_index, columns=[column_name])
        
        self.pattern_list.drop(self.pattern_list.loc[:self.t-self.window*7].index, inplace=True)
        self.pattern_list = pattern_list.combine_first(self.pattern_list)
        
        sample = []
        
        for index in pattern_list.index:
            for data in self.data:
                sample.append(data.loc[index:index+self.window])
        
        if not sample:
            return None
        else:
            return sample