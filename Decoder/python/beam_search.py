from collections import namedtuple
from typing import List
import scipy.sparse
import numpy as np
import itertools
from time import time
from numpy import dot
from multiprocessing import Pool
from functools import partial
from multiprocessing import Process, Pipe

class Token:
    def __init__(self, id, prev_id, arc_number, model_score, lm_score,creating_frame_num=-1):
        self.id = id
        self.prev_id = prev_id
        self.arc_number = arc_number
        self.model_score = model_score
        self.lm_score = lm_score
        self.creating_frame_num = creating_frame_num

class BeamSearch:
    """This class encapsulates the token lifecyclewhich makes the decoder class cleaner.
    """

    def __init__(self, beam_width):
        # the tokens array holds all of the tokens we've committed to the search
        self.tokens = []

        # set beam_witdht
        assert beam_width > 0
        self.beam_width = beam_width

    def set_intial_active_token(self, token):
        self.active_tokens = self.tokens =[token]
        self.last_token_id = self.active_tokens[0].id

    def token_list_to_sparse(self, num_arcs: int, token_list: List[Token]):
        """This helper function takes a list of tokens, and turns it into an equivalent set of sparse structures.
        """
        # token's score
        # the am score and lm score for an active arc is the path score from the begining of serach
        tokens_scores = np.array([token.model_score + token.lm_score for token in token_list], dtype=np.float32)
        # limit scores between 0->1 this very important as pathcost will weight the next transition cost limmiting numbers betwee 0->1 will prevent overflow and maintain relative difference
        tokens_scores = np.exp(tokens_scores - np.max(tokens_scores))
        # make a column vector; row index is arc number
        rows = np.array(
            [token.arc_number for token in token_list], dtype=np.int)
        cols = np.zeros(rows.shape)
        scores = scipy.sparse.csc_matrix(
            (tokens_scores, (rows, cols)),
            shape=(num_arcs, 1),
            dtype=np.float32
        )

        # try to change to dict mappng the arc number to token directly ____!!!!!
        score_index_to_token_index = np.ones(num_arcs, dtype=np.int32) * -1  # bogus initialization
        for i, token in enumerate(token_list):
            score_index_to_token_index[token.arc_number] = i

        return scores, score_index_to_token_index

    def get_norm_tokens_scores(self,tokens):
        tokens_scores = np.array([token.model_score + token.lm_score for token in tokens], dtype=np.float32)
        return np.exp(tokens_scores - np.max(tokens_scores)) #normalize score value beteween 0->1

    
    def beam_prune(self):
        if len(self.active_tokens) > self.beam_width:
            self.active_tokens = sorted(self.active_tokens, key=lambda x: x.model_score + x.lm_score, reverse=True)[0:self.beam_width]

    def commit_active_tokens(self):
        self.tokens += self.active_tokens

    def advance_token(self, prev_token: Token, new_token_arc, model_score, lm_score,creating_frame_num):
        self.last_token_id += 1
        return Token(
            self.last_token_id,
            prev_token.id,
            new_token_arc,
            prev_token.model_score + model_score,
            prev_token.lm_score + lm_score,
            creating_frame_num)
    
    def tok_backtrace(self):
        """
            This function finds the best path described by the tokens created so far.
            and asummes that the best final token exist at end of tokens list
        """
        looking_for_token_id = self.tokens[-1].id

        path = []
        # search backward through tokens
        for token in self.tokens[::-1]:
            if token.id == looking_for_token_id:
                arc_number = token.arc_number
                path.append((arc_number,token.creating_frame_num))
                looking_for_token_id = token.prev_id
        # reverse backtrace so tokens are in forward-time-order
        path = path[::-1]

        # Combine sequences of identical arcs into one representative arc_number removing self loops
        return [list(x)[-1] for k, x in itertools.groupby(path,lambda e:e[0])]
   
    def apply_final_state(self,arcs,final):
        dest_states = [arcs[token.arc_number].target_state for token in self.active_tokens]
        self.active_tokens = [
            Token(token.id, token.prev_id, token.arc_number,
                  token.model_score, token.lm_score + final[final_state],creating_frame_num=-1)
            for token, final_state in zip(self.active_tokens, dest_states) if final_state in final
        ]
        
        beest_final_token = max(self.active_tokens, key=lambda x: x.model_score + x.lm_score)
        self.active_tokens = [beest_final_token]
        self.commit_active_tokens()
        return beest_final_token

    def do_forward(self, trans_mat,obs_vector=None, arcs=None, all_trans=None, token_list=None,creating_frame_num=-1):
        """Implements the search-update algorithm using sparse matrix-vector primitives
        """
        new_tokens = []
        token_list = token_list if token_list is not None else self.active_tokens

        #  Convert the token list into a sparse structure
        src_score, src_token_index = self.token_list_to_sparse(len(arcs), token_list)

        # Project the tokens forward through the given transition matrix. Note that this is not
        trans = trans_mat.multiply(src_score.T)

        # Convert the sparse trans matrix into two obects: row_to_column, which for every row of
        row_to_column = np.array(trans.argmax(axis=1)).squeeze()
        active_rows = trans.max(axis=1).nonzero()[0] #nonzero return the rows indices that have maximums but not zero (paired with useless array so we use [0] to discard that array)

        new_tokens = [
            self.advance_token(
                    token_list[src_token_index[row_to_column[active_row]]],
                    active_row,
                    obs_vector[arcs[active_row].input_label_indx] if obs_vector is not None else 0,
                    all_trans[active_row, row_to_column[active_row]],
                    creating_frame_num
                )
                for active_row in active_rows]

        return new_tokens