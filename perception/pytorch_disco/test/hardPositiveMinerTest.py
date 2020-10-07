import os 
os.environ["MODE"] = "NEL_STA"
os.environ["exp_name"] = "trainer_big_builder_hard_exp5_pret"
os.environ["run_name"] = "check"

import numpy as np 
import hardPositiveMiner
import torch
import ipdb 
from DoublePool import DoublePool_O_f

st = ipdb.set_trace

def testEmbeddingPool():
    hpm = hardPositiveMiner.EmbeddingsPool(2)
    embedding = np.random.randint(0, 10, (10,3))
    hpm.update(embedding)
    print(hpm.fetch())
    assert hpm.is_full() == True, "Embedding pool must have been full"
    print(hpm.is_full())

def testHardPositiveMiner():
    hpm = hardPositiveMiner.HardPositiveMiner(2)
    embed = torch.from_numpy(np.random.randint(0,10,(3,1,3,20,30,35)))
    hpm.addToPool(embed)
    hpm.RandomQueryFeat()


def testResize():
    embed = torch.from_numpy(np.random.randint(0,10,(3,1,3,20,30,35)))
    hpm = hardPositiveMiner.HardPositiveMiner(2)
    hpm.resizeEmbedToRandomScale(embed)

def testExtractPatches():
    poole = DoublePool_O_f(10)
    poolg = DoublePool_O_f(10)
    embedding = np.random.randint(0,10,(10,1,4,10,15,20))
    filenames = ['f1']*10
    poole.update(embedding, embedding, embedding, filenames)
    poolg.update(embedding, embedding, embedding, filenames)
    hpm = hardPositiveMiner.HardPositiveMiner(2)
    rank = np.random.randint(0, 10, (10, 5))
    # featquery --> torch.Size([10, 4, 2, 2, 2])
    featquery, perm = hpm.extractPatches(poole)
    st()
    hpm.RetrievalRes(poole, rank, poolg, featquery, perm)

def testCosineSimilarity():
    kernel = torch.from_numpy(np.random.randint(0,10,(1,1,2,2,2))).float()
    feat = torch.from_numpy(np.random.randint(0,10,(1,1,5,5,5))).float()
    feat[:,:,2:4,2:4,2:4] = kernel
    variableAllOne =  torch.ones(1, feat.size()[1], feat.size()[2], feat.size()[3], feat.size()[4]).float()
    st()
    hpm = hardPositiveMiner.HardPositiveMiner()
    score = hpm.CosineSimilarity(feat, kernel, variableAllOne)
    st()

def testIndexCalculation():
    D = 5
    H = 7
    W = 11
    for i in range(385):
        index = i
        od = index//(H*W)
        index -= (od)*(H*W)
        oh = index//W
        ow = index%W
        
        index = i
        nd, nh, nw = index // (H*W), (index % (H*W))//W, index % (W)

        if od != nd or oh != nh or ow != nw:
            print("WRONG!!")
    
def testEverything():
    # gives red patches
    # get patches from poole tensors
    poole = DoublePool_O_f(10)
    poolg = DoublePool_O_f(10)
    embedding = torch.from_numpy(np.random.randn(10,1,4,10,15,20)).to(torch.device("cuda")).float()
    # embedding = torch.from_numpy(embedding).cuda()
    filenames = ['f1']*10
    poole.update(embedding, embedding, embedding, filenames)
    poolg.update(embedding, embedding, embedding, filenames)

    hpm = hardPositiveMiner.HardPositiveMiner()
    featquery, perm = hpm.extractPatches(poole)    
    rank = np.random.randint(0, 10, (10, 5))
    topkImg, topkScale, topkValue, topkW, topkH , topkD , topPoolgFnames, fname_e = hpm.RetrievalRes(poole, rank, poolg, featquery, perm)
    # topkImg is the indexes of ranks[perm] --> so the 3d tensors used are  emb_e[ranks[perm][topkImg]]
    minVal = 6 
    margin = 5
    searchRegion, validRegion = (2,10)
    nbImgEpoch = 10
    batchSize = 2

    iterEpoch = nbImgEpoch // batchSize
    posPair, fnamesForDataLoader = hpm.TrainPair(nbImgEpoch,topkImg, topkScale, topkW, topkH,  topkD,  margin,  searchRegion, validRegion,poolg,poole,rank, topPoolgFnames)

    posPairEpochIndex = hpm.DataShuffle(posPair, batchSize)

    embedding_final = torch.from_numpy(np.random.randn(batchSize,2,4,20,20,20)).cuda().float()
    for j_ in range(iterEpoch) :
        posSimilarityBatch = []        
        for k_ in range(batchSize):
            embedding_final_example= embedding_final[k_:k_+1]
            posSimilarity = hpm.PosSimilarity(embedding_final_example, posPair, posPairEpochIndex[j_, k_], topkImg, topkScale, topkD, topkH, topkW)
            posSimilarityBatch = posSimilarityBatch + posSimilarity
        posSimilarityBatch = torch.cat(posSimilarityBatch, dim=0)

    print("out")

def testEndToEnd():
    poole = DoublePool_O_f(3)
    poolg = DoublePool_O_f(3)
    embedding = torch.from_numpy(np.random.randn(3,1,1,15,15,15)).to(torch.device("cpu")).float()
    # embedding = torch.from_numpy(embedding).cuda()
    filenames = ['f1']*10
    poole.update(embedding, embedding, embedding, filenames)
    poolg.update(embedding, embedding, embedding, filenames)

    hpm = hardPositiveMiner.HardPositiveMiner()
    featquery, perm = hpm.extractPatches(poole)  
    st()

def compute_patch_based_scores(pool_e, pool_g, num_embeds):
    hpm = hardPositiveMiner.HardPositiveMiner()

    num_patches_per_emb = 10
    scores = torch.zeros((num_embeds, num_embeds)).cpu()
    '''
    This will create a dummy rank matrix which will look like this:
    0 1 2 ... num_embeds (1st row)
    0 1 2 ... num_embeds (2nd row)
    .
    .
    0 1 2 ... num_embeds  (num_embeds th row)
    '''
    dummy_ranks, _ = np.meshgrid(np.arange(num_embeds), np.arange(num_embeds))
    # _, dummy_ranks = torch.meshgrid(torch.arange(num_embeds),torch.arange(num_embeds))
    # dummy_ranks = dummy_ranks.cuda()
    for i in range(num_patches_per_emb):
        featQuery_i , perm_i =  hpm.extractPatches(pool_e)
        topkImg_i, _, topkValue_i, _, _, _  = hpm.RetrievalResForExpectation(pool_e, pool_g, featQuery_i)
        for j in range(topkImg_i.shape[0]):
            scores[j, topkImg_i[j].long()] += topkValue_i[j].cpu()
    st()
    return scores

def test_compute_patch_based_scores():
    poole = DoublePool_O_f(3)
    poolg = DoublePool_O_f(3)
    embedding = torch.from_numpy(np.random.randn(3,1,15,15,15)).to(torch.device("cpu")).float()
    # embedding = torch.from_numpy(embedding).cuda()
    filenames = ['f1']*10
    poole.update(embedding, embedding, embedding, filenames)
    poolg.update(embedding, embedding, embedding, filenames)
    compute_patch_based_scores(poole, poolg, embedding.shape[0])

def testRetrievalResForNegativeSim():
    poole = DoublePool_O_f(10)
    poolg = DoublePool_O_f(10)
    embedding = torch.from_numpy(np.random.randint(0,10,(10,4,10,15,20)))
    filenames = ['f1']*10
    poole.update(embedding, embedding, embedding, filenames)
    poolg.update(embedding, embedding, embedding, filenames)
    hpm = hardPositiveMiner.HardPositiveMiner()
    rank = np.random.randint(0, 10, (10, 5))
    # featquery --> torch.Size([10, 4, 2, 2, 2])
    featquery, perm = hpm.extractPatches(poole)
    hpm.RetrievalRes(poole, rank, poolg, featquery, perm, negativeSamples=True)


if __name__ == '__main__':
    testRetrievalResForNegativeSim()


