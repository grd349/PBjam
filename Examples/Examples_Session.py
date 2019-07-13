
# coding: utf-8

# In[1]:


import pbjam as pb
import pandas as pd
import lightkurve as lk
from tqdm import tqdm


# In[2]:


df = pd.read_csv('mytgts.csv')
df = df[:3]
df.head()


# In[3]:


sess = pb.session(dictlike=df, make_plots=True, nthreads=4,
                  use_cached=True, model_type='simple')


# In[4]:


sess()
