import numpy as np 
import hardPositiveMiner
import torch
import ipdb 
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
    st()


def testResize():
    embed = torch.from_numpy(np.random.randint(0,10,(3,1,3,20,30,35)))
    hpm = hardPositiveMiner.HardPositiveMiner(2)
    hpm.resizeEmbedToRandomScale(embed)

def test_TrainPair():
    # number of patches used for finding correspondences.. in our case this is the len of emb_e
    hpm = hardPositiveMiner.HardPositiveMiner(3)
    num_es = 3
    top_match = 10
    searchImgList = [str(i) for i in range(num_es)]
    nbPatchTotal, searchDir = (num_es,"../data/Brueghel/")
    searchImgList, topkImg  = (searchImgList,np.random.randint(0,num_es,(num_es,top_match)))
    D,H,W = (20,30,35)
    minVal = 6 
    batchSize = 3
    transform, net, margin = (None,None,5)
    cuda, featChannel = (True,3)
    searchRegion, validRegion = (2,10)
    nbImgEpoch, minNet, strideNet  = (num_es,15,16)
    embed = torch.from_numpy(np.random.randint(0,10,(num_es,1,featChannel,D,H,W))).cuda()
    topkScale, topkW, topkH, topkD = (np.random.randint(35,40,(num_es,top_match)),np.random.randint(minVal,W-minVal,(num_es,top_match)),np.random.randint(minVal,H-minVal,(num_es,top_match)),np.random.randint(minVal,D-minVal,(num_es,top_match)))
    hpm.addToPool(embed,None,searchImgList)
    embed_e,img_e,file_e = hpm.pool.fetch()
    embed_g,img_g,file_g = hpm.pool.fetch()
    posPair, _ = hpm.TrainPair(topkImg, topkScale, topkW, topkH, topkD, margin, cuda, featChannel, searchRegion, validRegion, nbImgEpoch, minNet, strideNet,embed_e,embed_g,file_e,file_g)
    posPairEpoch = hpm.DataShuffle(posPair, batchSize)
    
    st()
    print("done")
    
if __name__ == '__main__':
    # testHardPositiveMiner()
    test_TrainPair()

    # inf = InfiniteSampler()
    # st()
    # for i,data in inf.loop():
    #     print(data)
    

