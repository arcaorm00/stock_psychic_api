import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
basedir = os.path.abspath(os.path.dirname(__file__))
from com_stock_api.utils.file_helper import FileReader
import pandas as pd
import numpy as np
import uuid

'''
RowNumber,        레코드 행 번호
CustomerId,       고객 ID
Surname,          고객 last name
CreditScore,      신용점수
Geography,        지역
Gender,           성별
Age,              나이
Tenure,           존속 기간
Balance,          잔액
NumOfProducts,    구매 물품 수
HasCrCard,        신용카드 여부
IsActiveMember,   활성 고객 여부
EstimatedSalary,  급여 수준
Exited            서비스 탈퇴 여부
'''

# ---------------------- raw data를 database 저장용 데이터로 전환
class MemberDBDataProcessing:

    def __init__(self):
        print(f'basedir: {basedir}')
        self.fileReader = FileReader()
        self.datapath = os.path.join(basedir, 'data')

    def process(self, data):
        service = self
        this = self.fileReader
        this.context = self.datapath
        this.train = service.new_model(data)
        # print(f'feature 드롭 전 변수: \n{this.train.columns}')
        this = service.age_ordinal(this)
        # print(f'나이 정제 후: \n{this.train.head()}')
        this = service.estimatedSalary_ordinal(this)
        # print(f'수입 정제 후: \n{this.train.head()}')
        this = service.password_nominal(this)
        # print(f'비밀번호 추가 후: \n{this.train["Password"]}')
        this = service.email_nominal(this)
        # print(f'이메일 추가 후: \n{this.train["Email"]}')
        this = service.role_nominal(this)
        # print(f'권한 추가 후: \n{this.train["Role"]}')
        self.datapath = os.path.join(basedir, 'saved_data')
        this.train.to_csv(os.path.join(self.datapath, 'member_detail.csv'), index=False)
        return this

    def new_model(self, payload) -> object:
        this = self.fileReader
        this.context = self.datapath
        this.fname = payload
        print(f'*****{this.context + this.fname}')
        return pd.read_csv(os.path.join(self.datapath, this.fname))

    @staticmethod
    def drop_feature(this, feature) -> object:
        this.train = this.train.drop([feature], axis=1)
        return this

    @staticmethod
    def surname_nominal(this):
        return this

    @staticmethod
    def creditScore_ordinal(this):
        return this

    @staticmethod
    def geography_nominal(this):
        return this

    @staticmethod
    def gender_nominal(this):
        return this

    @staticmethod
    def age_ordinal(this):
        train = this.train
        train['Age'] = train['Age'].fillna(-0.5)
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf] # 범위
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'YoungAdult', 'Adult', 'Senior']
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        age_title_mapping = {
            0: 'Unknown',
            1: 'Baby', 
            2: 'Child',
            3: 'Teenager',
            4: 'Student',
            5: 'YoungAdult',
            6: 'Adult',
            7: 'Senior'
        }

        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown':
                train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]
        
        age_mapping = {
            'Unknown': 0,
            'Baby': 1, 
            'Child': 2,
            'Teenager': 3,
            'Student': 4,
            'YoungAdult': 5,
            'Adult': 6,
            'Senior': 7
        }
        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        this.train = train
        return this

    @staticmethod
    def tenure_ordinal(this):
        return this

    @staticmethod
    def balance_ordinal(this):
        return this

    @staticmethod
    def numOfProducts_ordinal(this):
        return this

    @staticmethod
    def hasCrCard_numeric(this):
        return this

    @staticmethod
    def isActiveMember_numeric(this):
        return this

    @staticmethod
    def estimatedSalary_ordinal(this):
        this.train['EstimatedSalary'] = pd.qcut(this.train['EstimatedSalary'], 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this

    # 비밀번호 추가 (임시 1234 통일)
    @staticmethod
    def password_nominal(this):
        this.train['Password'] = '1234'
        return this

    # 이메일 추가 (임시 uuid_CustomerId@gmail.com)
    @staticmethod
    def email_nominal(this):
        this.train['Email'] = ''
        for idx in range(len(this.train)):
            this.train.loc[idx,'Email'] = str(uuid.uuid1()).split('-')[3] + '_' + str(this.train.loc[idx,'CustomerId']) + '@gmail.com'
        return this
    
    # 권한 추가 (모두 회원 권한)
    @staticmethod
    def role_nominal(this):
        this.train['Role'] = 'ROLE_USER'
        return this
    


# ---------------------- database 저장 파일을 모델링용 데이터로 전환
class MemberModelingDataPreprocessing:
    
    def __init__(self):
        self.filereader = FileReader()

    def hook_process(self):
        this = self.filereader
        this.context = os.path.join(basedir, 'saved_data')
        # 데이터 정제 전 database data
        this.fname = 'member_detail.csv'
        members = this.csv_to_dframe()
        this.train = members
        
        # 컬럼 삭제
        this = self.drop_feature(this, 'RowNumber') # 열 번호 삭제
        this = self.drop_feature(this, 'Surname') # 이름 삭제
        this = self.drop_feature(this, 'Email') # 이메일 삭제
        this = self.drop_feature(this, 'Role') # 권한 삭제
        this = self.drop_feature(this, 'Password') # 비밀번호 삭제
        
        # 데이터 정제
        this = self.geography_nominal(this)
        this = self.gender_nominal(this)
        this = self.age_ordinal(this)
        this = self.drop_feature(this, 'Age')
        this = self.creditScore_ordinal(this)
        this = self.balance_ordinal(this)
        this = self.estimatedSalary_ordinal(this)

        # 고객의 서비스 이탈과 각 칼럼간의 상관계수
        self.correlation_member_secession(this.train)

        # label 컬럼 재배치
        this = self.columns_relocation(this)

        # 정제 데이터 저장
        self.save_preprocessed_data(this)

        # 훈련 데이터, 레이블 데이터 분리
        # this.label = self.create_label(this)
        # this.train = self.create_train(this)
        
        # print(this)
        

    # 고객의 서비스 이탈과 각 칼럼간의 상관계수
    def correlation_member_secession(self, members):
        member_columns = members.columns
        member_correlation = {}
        for col in member_columns:
            cor = np.corrcoef(members[col], members['Exited'])
            # print(cor)
            member_correlation[col] = cor
        print(member_correlation)
        '''
        r이 -1.0과 -0.7 사이이면, 강한 음적 선형관계,
        r이 -0.7과 -0.3 사이이면, 뚜렷한 음적 선형관계,
        r이 -0.3과 -0.1 사이이면, 약한 음적 선형관계,
        r이 -0.1과 +0.1 사이이면, 거의 무시될 수 있는 선형관계,
        r이 +0.1과 +0.3 사이이면, 약한 양적 선형관계,
        r이 +0.3과 +0.7 사이이면, 뚜렷한 양적 선형관계,
        r이 +0.7과 +1.0 사이이면, 강한 양적 선형관계

        result:

        {'CustomerId': array([[ 1.        , -0.00624799], [-0.00624799,  1.        ]]),  ==> 거의 무시될 수 있는 선형관계
        'CreditScore': array([[ 1.        , -0.02709354], [-0.02709354,  1.        ]]), ==> 거의 무시될 수 있는 선형관계
        'Geography': array([[1.        , 0.15377058], [0.15377058, 1.        ]]), ==> 약한 양적 선형관계
        'Gender': array([[1.        , 0.10651249], [0.10651249, 1.        ]]), ==> 약한 양적 선형관계
        'Age': array([[1.        , 0.28532304], [0.28532304, 1.        ]]), ==> 약한 양적 선형관계
        'Tenure': array([[ 1.        , -0.01400061], [-0.01400061,  1.        ]]), ==> 거의 무시될 수 있는 선형관계
        'Balance': array([[1.        , 0.11853277], [0.11853277, 1.        ]]), ==> 약한 양적 선형관계
        'NumOfProducts': array([[ 1.        , -0.04781986], [-0.04781986,  1.        ]]),  ==> 거의 무시될 수 있는 선형관계
        'HasCrCard': array([[ 1.        , -0.00713777], [-0.00713777,  1.        ]]),  ==> 거의 무시될 수 있는 선형관계
        'IsActiveMember': array([[ 1.        , -0.15612828], [-0.15612828,  1.        ]]), ==> 약한 음적 선형관계
        'EstimatedSalary': array([[1.        , 0.01300995], [0.01300995, 1.        ]]),  ==> 거의 무시될 수 있는 선형관계
        'Exited': array([[1., 1.], [1., 1.]]), 
        'AgeGroup': array([[1.        , 0.21620629], [0.21620629, 1.        ]])} ==> 약한 양적 선형관계
        '''

    # ---------------------- 데이터 정제 ----------------------
    @staticmethod
    def create_train(this):
        return this.train.drop('Exited', axis=1)

    @staticmethod
    def create_label(this):
        return this.train['Exited']

    @staticmethod
    def drop_feature(this, feature) -> object:
        this.train = this.train.drop([feature], axis=1)
        return this

    @staticmethod
    def surname_nominal(this):
        return this

    @staticmethod
    def creditScore_ordinal(this):
        this.train['CreditScore'] = pd.qcut(this.train['CreditScore'], 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this

    @staticmethod
    def geography_nominal(this):
        # print(this.train['Geography'].unique()) 
        # ==> ['France' 'Spain' 'Germany']
        geography_mapping = {'France': 1, 'Spain': 2, 'Germany': 3}
        this.train['Geography'] = this.train['Geography'].map(geography_mapping)
        return this

    @staticmethod
    def gender_nominal(this):
        gender_mapping = {'Male': 0, 'Female': 1}
        this.train['Gender'] = this.train['Gender'].map(gender_mapping)
        this.train = this.train
        return this

    @staticmethod
    def age_ordinal(this):
        train = this.train
        train['Age'] = train['Age'].fillna(-0.5)
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf] # 범위
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'YoungAdult', 'Adult', 'Senior']
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        age_title_mapping = {
            0: 'Unknown',
            1: 'Baby', 
            2: 'Child',
            3: 'Teenager',
            4: 'Student',
            5: 'YoungAdult',
            6: 'Adult',
            7: 'Senior'
        }

        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown':
                train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]
        
        age_mapping = {
            'Unknown': 0,
            'Baby': 1, 
            'Child': 2,
            'Teenager': 3,
            'Student': 4,
            'YoungAdult': 5,
            'Adult': 6,
            'Senior': 7
        }
        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        this.train = train
        return this

    @staticmethod
    def tenure_ordinal(this):
        return this

    @staticmethod
    def balance_ordinal(this):
        this.train['Balance'] = pd.qcut(this.train['Balance'].rank(method='first'), 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this

    @staticmethod
    def numOfProducts_ordinal(this):
        return this

    @staticmethod
    def hasCrCard_numeric(this):
        return this

    @staticmethod
    def isActiveMember_numeric(this):
        return this

    @staticmethod
    def estimatedSalary_ordinal(this):
        this.train['EstimatedSalary'] = pd.qcut(this.train['EstimatedSalary'], 10, labels={1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
        return this

    # ---------------------- 파일 저장 ---------------------- 
    def save_preprocessed_data(self, this):
        this.context = os.path.join(basedir, 'saved_data')
        this.train.to_csv(os.path.join(this.context, 'member_refined.csv'))

    # ---------------------- label 컬럼 위치 조정 ---------------------- 
    def columns_relocation(self, this):
        cols = this.train.columns.tolist()
        # ['CustomerId', 'CreditScore', ... , 'EstimatedSalary', 'Exited', 'AgeGroup']
        cols =  (cols[:-2] + cols[-1:]) + cols[-2:-1]
        # ['CustomerId', 'CreditScore', ... , 'EstimatedSalary', 'AgeGroup', 'Exited']
        this.train = this.train[cols]
        return this


# ---------------------- MemberPro ---------------------- 
class MemberPro:
    def __init__(self):
        self.db_data_process = MemberDBDataProcessing()
        self.modeling_data_process = MemberModelingDataPreprocessing()

    def hook(self):
        ddp = self.db_data_process
        ddp.process('member_dataset.csv')
        mdp = self.modeling_data_process
        mdp.hook_process()
    

if __name__ == '__main__':
    mp = MemberPro()
    mp.hook()
    