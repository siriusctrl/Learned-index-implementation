from BloomFilter import BloomFilter
import math
import random
import mmh3

class HybridBloomFilter():
    
    def __init__(self, model, data, fp_rate):
        self.model = model
        self.threshold = None
        self.fp_rate = float(fp_rate)
        self.fit(data)
        self.create_bloom_filter(data)
        