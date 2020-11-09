import gensim,logging

logging.basicConfig(format='%(asctime)s: %(levelname)s : %(message)s',level=logging.INFO)
sentencs = [['first','sentence'],['second','sentence']]

model = gensim.models.Word2Vec(sentencs,min_count=1)
print(model['sentence'])