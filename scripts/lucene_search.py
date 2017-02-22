'''
Use Lucene to retrieve candidate documents for given a query.
'''
import shutil
import os
import lucene
import parameters as prm
import utils
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, DirectoryReader, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory, NIOFSDirectory, MMapDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
import time
from collections import OrderedDict, defaultdict
from multiprocessing.pool import ThreadPool
import Queue
import math
from nltk.tokenize import wordpunct_tokenize

class LuceneSearch():

    def __init__(self, docs):

        self.env = lucene.initVM(initialheap='28g', maxheap='28g', vmargs=['-Djava.awt.headless=true'])
        if not os.path.exists(prm.index_folder):
            print 'Creating index at', prm.index_folder
            self.create_index(docs)
        
        self.index_folder = prm.index_folder

        fsDir = MMapDirectory(Paths.get(self.index_folder))
        self.searcher = IndexSearcher(DirectoryReader.open(fsDir))
        self.analyzer = StandardAnalyzer()

        self.pool = ThreadPool(processes=prm.n_threads)


    def create_index(self, docs):

        os.mkdir(prm.index_folder)

        t1 = FieldType()
        t1.setStored(True)
        #t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

        t2 = FieldType()
        t2.setStored(False)
        t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

        indexDir = MMapDirectory(Paths.get(prm.index_folder))
        writerConfig = IndexWriterConfig(StandardAnalyzer())
        writer = IndexWriter(indexDir, writerConfig)
        print "%d docs in index" % writer.numDocs()
        print "Indexing documents..."
       
        
        n = 0
        for doc_id, txt in docs.items():
            doc = Document()
            txt_ = txt.lower()

            doc.add(Field("id", str(doc_id), t1))
            doc.add(Field("text", txt, t2))

            writer.addDocument(doc)
            n += 1
            if n % 1000 == 0:
                print 'indexing article', n
        print "Indexed %d docs from reference files (%d docs in index)" % (n, writer.numDocs())
        print "Closing index of %d docs..." % writer.numDocs()
        writer.close()


    def search_multithread(self, qs, max_cand):

        self.max_cand = max_cand
        out = self.pool.map(self.search_multithread_part, qs)
 
        return out


    def search_multithread_part(self, q):

        if not self.env.isCurrentThreadAttached():
            self.env.attachCurrentThread()

        try:
            q = q.replace('AND','\\AND').replace('OR','\\OR').replace('NOT','\\NOT')
            query = QueryParser("text", self.analyzer).parse(QueryParser.escape(q))
        except:
            print 'Unexpected error when processing query:', str(q)
            print 'Using query "dummy".'
            q = 'dummy'
            query = QueryParser("text", self.analyzer).parse(QueryParser.escape(q))

        c = OrderedDict()
        hits = self.searcher.search(query, self.max_cand)

        for hit in hits.scoreDocs:
            doc = self.searcher.doc(hit.doc)
            c[int(doc['id'])] = doc['text']

        return c

    
    def search_singlethread(self, qs, max_cand):

        out = []
        for q in qs:
            try:
                q = q.replace('AND','\\AND').replace('OR','\\OR').replace('NOT','\\NOT')
                query = QueryParser("text", self.analyzer).parse(QueryParser.escape(q))
            except:
                print 'Unexpected error when processing query:', str(q)
                print 'Using query "dummy".'
                query = QueryParser("text", self.analyzer).parse(QueryParser.escape('dummy'))

            c = OrderedDict()
            hits = self.searcher.search(query, max_cand)

            for hit in hits.scoreDocs:
                doc = self.searcher.doc(hit.doc)
                c[int(doc['id'])] = doc['text']

            out.append(c)

        return out


    def get_candidates(self, qs, max_cand):
        if prm.n_threads > 1:
            return self.search_multithread(qs, max_cand)
        else:
            return self.search_singlethread(qs, max_cand)


