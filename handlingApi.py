import requests
from xml.etree import ElementTree
import os


def gatherData(isbnId):
	payload = {'key': '5ZzjDdTbEpwuW1odFWpuWw'}
	res = requests.get('https://www.goodreads.com/book/isbn/' + isbnId, params = payload);
	with open('xmlFile.xml', 'w') as w:
		w.write(res.text)
	file = 'xmlFile.xml'
	fullPath = os.path.abspath(os.path.join(file))
	dom = ElementTree.parse(fullPath)
	reviews = dom.findall('book/description')

	
	for r in reviews:
		rawData = r.text

	with open('rawData.txt', 'w') as w:
		w.write(rawData)
	return rawData

