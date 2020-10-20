from flask_restful import Resource, reqparse
from com_stock_api.korea_covid.dao import KoreaDao
from com_stock_api.korea_covid.dto import KoreaDto

class KoreaCovid(Resource):

    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('date', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('seoul_cases', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('seoul_death', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('total_cases', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('total_death', type=int, required=False, help='This field cannot be left blank')

    def post(self):
        data = self.parset.parse_args()
        koreacovid =KoreaDto(data['date'],data['seoul_cases'],data['seoul_cases'],data['total_cases'],data['total_death'])
        
        try:
            koreacovid.save()
        except:
            return {'message':'An error occured inserting the koreacovid'}, 500
        return koreacovid.json(), 201

    def get(self,id):
        koreacovid = KoreaDao.find_by_id(id)
        if koreacovid:
            return koreacovid.json()
        return {'message': 'koreacovid not found'}, 404

    def put (self, id):
        data = KoreaCovid.parser.parse_args()
        koreacovid = KoreaDto.find_by_id(id)

        koreacovid.id =data['id']
        koreacovid.seoul_cases = data['seoul_cases']
        koreacovid.seoul_death= data['seoul_death']
        koreacovid.total_cases = data['total_cases']
        koreacovid.total_death= data['total_deatb']
        koreacovid.save()
        return koreacovid.json()

class KoreaCovids(Resource):
    def get(self):
        return {'koreacovids': list(map(lambda koreacovid: koreacovid.json(), KoreaDao.find_all()))}
        #return {'kospis':[kospi.json() for kospi in KospiDao.find_all()]}

