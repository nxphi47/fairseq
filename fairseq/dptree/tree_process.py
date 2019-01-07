from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
from nltk import treetransforms
from copy import deepcopy
from collections import deque
import queue
import argparse

from nltk.tree import Tree


# from bllipparser import RerankingParser
# import pickle

def padding_leaves(tree):
    leaves_location = [tree.leaf_treeposition(i) for i in range(len(tree.leaves()))]
    for i in range(len(leaves_location)):
        tree[leaves_location[i]] = "{0:03}".format(i) + "||||" + tree[leaves_location[i]]
    for i in range(len(tree.leaves())):
        if len(tree[tree.leaf_treeposition(i)[:-1]]) > 1:
            tree[tree.leaf_treeposition(i)] = Tree(tree[tree.leaf_treeposition(i)[:-1]].label(), [tree.leaves()[i]])


def bft(tree):
    meta = dict()
    list_subtree = list(tree.subtrees())
    lst_tree = []
    lst = []
    queue_tree = queue.Queue()
    queue_tree.put(tree)
    meta[list_subtree.index(tree)] = []
    while not queue_tree.empty():
        node = queue_tree.get()
        lst.append(node)
        lst_tree.append(meta[list_subtree.index(node)])
        for i in range(len(node)):
            child = node[i]
            if isinstance(child, nltk.Tree):
                meta[list_subtree.index(child)] = deepcopy(meta[list_subtree.index(node)])
                meta[list_subtree.index(child)].append(i)
                queue_tree.put(child)
    return lst, lst_tree, meta


def clean_node(tree):
    t3 = deepcopy(tree)
    t3_lst, t3_lst_tree, t3_meta = bft(t3)
    for ind, sub in reversed(list(enumerate(t3.subtrees()))):
        if sub.height() >= 2:
            postn = t3_meta[ind]
            parentpos = postn[:-1]
            if parentpos and len(t3[parentpos]) == 1:
                t3[parentpos] = t3[postn]
            # postn = parentpos
            # parentpos = postn[:-1]
    leaves_location = [t3.leaf_treeposition(i) for i in range(len(t3.leaves()))]
    for i in range(len(leaves_location)):
        t3[leaves_location[i]] = t3[leaves_location[i]][7:]
    if len(t3) == 1:
        t3 = t3[0]
    return t3


def generate_data(tree):
    cnfTree = deepcopy(tree)
    treetransforms.chomsky_normal_form(cnfTree)
    padding_leaves(cnfTree)
    bf_tree, bf_lst_tree, bf_meta = bft(cnfTree)
    # leaves_cnfTree = cnfTree.leaves()
    input_node = []
    input_label = []
    input_index = []
    leaves_location = [cnfTree.leaf_treeposition(i) for i in range(len(cnfTree.leaves()))]
    for i in range(len(bf_lst_tree)):
        if len(bf_tree[i].leaves()) > 1:
            if '|' in bf_tree[i].label():
                input_node.append("SPLIT_NODE_node_label")
                input_label.append("<pad>")
            else:
                input_node.append(bf_tree[i].label() + "_node_label")
                input_label.append('<pad>')
        else:
            input_label.append(bf_tree[i].label() + "_leaf_label")
            input_node.append(bf_tree[i].leaves()[0][7:])

        first_leaf = deepcopy(bf_lst_tree[i])
        first_leaf.extend(bf_tree[i].leaf_treeposition(0))
        first_leaf = leaves_location.index(tuple(first_leaf))
        last_leaf = first_leaf + len(bf_tree[i].leaves()) - 1
        input_index.append([first_leaf, last_leaf])
    return input_node, input_label, input_index


def tree2matrix(tree):
    cnfTree = deepcopy(tree)
    treetransforms.chomsky_normal_form(cnfTree)
    node_label = set([])
    leaf_label = set([])
    # print("The binary transform tree is ")
    # cnfTree.pretty_print()
    leaves = cnfTree.leaves()
    leaves_position = []
    for i in range(len(leaves)):
        leaves_position.append(cnfTree.leaf_treeposition(i))
    matrix = []
    for i in range(len(leaves)):
        list_i = ['<pad>'] * len(leaves)
        leaf_i = leaves_position[i]
        for k in range(len(leaf_i) - 1, -1, -1):
            if set(leaf_i[k:]) == set([0]):
                tree_at_k = cnfTree[leaf_i[:k]]
                label_k = tree_at_k.label()
                if k == len(leaf_i) - 1:
                    leaf_label.add(label_k + "_leaf_label")
                else:
                    node_label.add(label_k + "_node_label")
            list_i[i + len(tree_at_k.leaves()) - 1] = label_k
        matrix.append(list_i)
    node_label.add('<pad>')
    leaf_label.add('<pad>')
    node_label.remove('<pad>')
    leaf_label.remove('<pad>')
    return leaves, matrix, node_label, leaf_label


def text2code(vocabulary, node_data, label_data):
    node_ = []
    label_ = []
    for i in range(len(node_data)):
        code_node = [vocabulary[x] for x in node_data[i]]
        node_.append(code_node)
        code_label = [vocabulary[x] for x in label_data[i]]
        label_.append(code_label)
    return node_, label_
