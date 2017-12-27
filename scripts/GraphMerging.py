""" Core for graph based mergers

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 19th May, 2017
"""
import threading
import time
import TerminalColors
import numpy as np
tcol = TerminalColors.bcolors()

class Node:
    def __init__(self, uid, parent=None):
        self.uid = uid
        self.parent = parent

    def __repr__(self):
        return '(u=%-4d, g=%-4d)' %(self.uid,get_gid(self))


def get_gid( node, verbose=False ):
    while node.parent is not None:
        node = node.parent
    return node.uid




class NonSeqMergeThread( threading.Thread ):
    def __init__(self, threadID, name, all_nodes, internal_e, n_components, thread_shared_scalars, S_word ):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.all_nodes = all_nodes
        self.internal_e = internal_e
        self.n_components = n_components
        self.thread_shared_scalars = thread_shared_scalars
        self.S_word = S_word

    def thread_print( self, to_print ):
        print  tcol.OKBLUE, '[',self.name, ']', to_print, tcol.ENDC

    def run(self):
        print 'Starting ', self.name
        t = threading.currentThread()
        prev_len = 0
        while getattr( t, "do_run", True ):
            time.sleep(1)
            if len(self.all_nodes) != prev_len: #print only if changed
                prev_len = len(self.all_nodes)


                keys = []
                for k in sorted( self.internal_e ):
                    if k != self.thread_shared_scalars['active_set_id']:
                        keys.append(k)

                self.thread_print( 'keys %s' %( keys ) )
                if len(keys) > 2:
                    K = self.compute_segment_similarity( keys )
                    # print K
                    self.print_pairwise_similarity( K, keys )


                self.thread_print( 'S_word.shape : %d' %(len(self.S_word)) )
                self.thread_print( 'len(all_nodes) : %d' %(len(self.all_nodes) ) )
                self.thread_print( '# components : %d/%d' %(len(self.internal_e), len(self.n_components) ) )
                self.thread_print( 'active_id : %d' %(self.thread_shared_scalars['active_set_id']))
                for k in sorted( self.internal_e ):
                    self.thread_print( '%4d : %4d %2.3f' %(k, self.n_components[k], self.internal_e[k]) )
        print 'Ending ', self.name

    def stop(self):
        self.do_run = False


    ## Given a list of keyframes of segments return a every-pair similarity score
    def compute_segment_similarity(self, keys ):
        S_word = self.S_word
        return np.dot( [S_word[k] for k in keys] , np.transpose( [S_word[k] for k in keys] ) )

    def print_pairwise_similarity( self, K, keys ):
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                if i<j:
                    self.thread_print( '<%4d,%4d> = %4.4f' %(keys[i],keys[j],K[i,j]) )
