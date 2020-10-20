from flask_restful import Resource, reqparse
from com_stock_api.us_covid.us_covid_dto import USCovidDto
from com_stock_api.us_covid.us_covid_dao import USCovidDao

class USCovid(Resource):


    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('date', type=str, required=False, help='This field cannot be left blank')
        parser.add_argument('total_case', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('total_death', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('ca_cases', type=int, required=False, help='This field cannot be left blank')
        parser.add_argument('ca_death', type=int, required=False, help='This field cannot be left blank')


    
    def post(self):
        data = self.parset.parse_args()
        uscovid = USCovidDto(data['date'], data['total_case'], data['total_death'], data['ca_cases'], data['ca_death'])
        try: 
            uscovid.save()
        except:
            return {'message': 'An error occured inserting the covid case'}, 500
        return uscovid.json(), 201
    
    
    def get(self, id):
        uscovid = USCovidDao.find_by_id(id)
        if uscovid:
            return uscovid.json()
        return {'message': 'uscovid not found'}, 404

    def put(self, id):
        data = USCovid.parser.parse_args()
        uscovid = USCovidDao.find_by_id(id)

        uscovid.title = data['title']
        uscovid.content = data['content']
        uscovid.save()
        return uscovid.json()

class USCovids(Resource):
    def get(self):
        return {'uscovid': list(map(lambda article: article.json(), USCovidDao.find_all()))}
