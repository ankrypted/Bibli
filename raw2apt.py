import re
import handlingApi as api

def cleanhtml(isbnId):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', api.gatherData(isbnId))
  return cleantext

def review(isbnId):
	print(cleanhtml(isbnId))

