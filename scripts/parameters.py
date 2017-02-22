data_folder = '/scratch/rfn216/NeuralResearcher/'
n_threads = 20 # number of parallel process that will execute the queries on the search engine.
index_name = 'index_874' # index name for the search engine. Used when engine is 'lucene', 'elastic', or 'whoosh'.
index_folder = data_folder + '/data/' + index_name + '/' # folder to store lucene's index. It will be created in case it does not exist.
