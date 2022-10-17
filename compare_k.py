#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import read_data
import control_group
import prefix_sum
import utils
import numpy as np
import itertools
from attribute_module import attribute
import weighted_count_query
import weighted_frequency_query
from estimation import Synthesizer
from query_module import range_query
import query_module
import copy
import random
import math


def my_solution2(attributes, measure, epsilon, noisy, partition,g):
	datas = read_data.merge(attributes)
	# table=datas
	table = random.sample(datas, len(datas) // partition)
	len_list = np.ones(len(table), dtype=int)
	pfs = prefix_sum.prefixsum(attributes, measure, epsilon,g)
	pfs.collection(table, len_list, noisy)
	return pfs.result


def my_solution3(attributes,measure,epsilon, noisy, partition,g):
    datas=read_data.merge(attributes)
    #print("the attributes to be collected:")
    #print(datas)
    table=random.sample(datas, len(datas) // partition)
    len_list=np.ones(len(table),dtype=int)
    #print(table)
    pfs1=prefix_sum.prefixsum(attributes,measure,epsilon,g)
    pfs1.collection_with_olh(table, len_list, noisy)
    #print(pfs.result)
    return pfs1.result

if __name__ == '__main__':
	# path="E:/test.csv"
	# infopath="E:\info.csv"
	g=2 #PRISM中的划分粒度

	d=30

	kk_list=[0,int(d/4),int(d/2),int(3*d/4),d]
	# print(type(result))
	print('done.')
	n = 30
	k = 2
	epsilon = 2
	query_volume =0.1
	data_size = 10**5

	print('loading data...')
	# path = "E:\PyWorkSpace\program\dataset/data_adult_30-30-1M.csv"
	path="../dataset/data_ipums_30-"+str(d)+"-1M.csv"
	# infopath = "E:\PyWorkSpace\program\dataset/info_30-30.csv"
	infopath = "../dataset/info_30-"+str(d)+".csv"
	# path = "C:/Users\FlyFF\Desktop\学习\论文\program\dataset_high\Adult/adult_10_10_05.csv"
	# infopath = "C:/Users\FlyFF\Desktop\学习\论文\program\dataset_high\Adult/adult_info-10.csv"
	result = read_data.readcsv(path)
	result=result[0:data_size]
	result2 = copy.deepcopy(result)
	info = read_data.readinfo(infopath)


	# attris = ['MONTH', 'AGE']
	print(result[0])
	attris = []
	for i in range(30):
		attri = 'D' + str(i + 1)
		attris.append(attri)
	measure = 'M'
	# attris = ['A', 'B']
	# measure = 'V'
	# get orginal distribution
	domain = []
	attributes = []
	for attri in attris:
		new_attri = attribute(attri, result, info, 0)
		attributes.append(new_attri)
		domain.append(new_attri.domain)
	datas = read_data.merge(attributes)
	mea = attribute(measure, result, info, 0)

	# 后续修改
	#
	# print('computing true distribution...')
	# table_dict=weighted_frequency_query.convert_data(datas, mea)
	# #print(table_dict)
	# table ,xx = weighted_count_query.convert_data(datas, mea,1)
	# #print(table_dict)
	# original_data=np.zeros(domain, dtype=float)
	# for data in table:
	#     index=tuple(data)
	#     original_data[index]+=1
	# #print('original_data:',original_data)
	# original_data/=len(table)
	# print('done.')
	ROUNDS = 3
	MEAN_ERROR_PRISM = [0,0,0,0,0,0]
	# generate queries and raw answers
	for ii in range(len(kk_list)):
		kk=kk_list[ii]
		print('generating queries...')
		r_list = []
		for i in range(n):
			# r_list.append(query_module.get_range(domain,query_volume))
			r_list.append(query_module.get_range(domain, query_volume))
		query_list = []
		for r in r_list:
			# generate query
			new_query = range_query(attris, r, k, epsilon, data_size)
			# add raw answer for query
			q_domain = []
			q_attributes = []
			for attri in new_query.attributes:
				new_attri = attribute(attri, result, info, 0)
				q_attributes.append(new_attri)
				q_domain.append(new_attri.domain)
			datas = read_data.merge(q_attributes)
			marginal = np.zeros(q_domain, dtype=float)
			for data in datas:
				index = tuple(data)
				marginal[index] += 1
			marginal /= len(datas)
			new_query.add_answer(marginal)
			query_list.append(range_query(attris, r, k, epsilon, data_size))
		print('done.')

		print('executing PRISM...')
		ERROR_PRISM = 0
		tables_dict = weighted_frequency_query.spilt_table(result, mea)
		marginal_PRISM = {}
		for weight in tables_dict.keys():
			# 现有代码默认K为D的一半，有提升空间
			# kk = int(len(attris) / 2) + 1
			if kk>0:
				C = math.factorial(kk) // (math.factorial(2) * math.factorial(kk - 2))
			else:
				C=0
			partition = C + len(attris)
			print('collecting single dimensions for weight:', weight, '...')
			single_ps = {}
			sub_table = tables_dict[weight]
			for attri in attris:
				attributes = []
				attributes.append(attribute(attri, sub_table, info, 0))
				# 将1-way数据收集并存入single_ps
				single_ps[attri] = my_solution2(attributes, measure, epsilon, 1, partition,g)
			# attri_dict = {}
			i = 0
			print('selecting valuable dimensions...')
			# for key in single_ps.keys():
			#     attri_dict[key] = i
			#     i = i + 1
			value_ps = Synthesizer.value_attribute(single_ps)
			#print(value_ps)
			pairs = utils.get_pair(value_ps)
			single_list = single_ps.values()
			print('collecting joint dimensions...')
			double_dict = {}
			# attributes = []
			#print(pairs)
			for pair in pairs:
				attributes = []
				print(pair)
				pair_num = []
				for attri in pair:
					new_attri = attribute(attri, result, info, 0)
					attributes.append(new_attri)
					# pair_num.append(attri_dict[attri])
				# key_num = tuple(pair_num)
				double_ps = my_solution2(attributes, measure, epsilon, 1, partition,g)
				double_dict[pair] = double_ps
			print('estimating entire model...')
			ERROR_PRISM = 0
			for query in query_list:
				attri_dict = {}
				i = 0
				selected_domain = []
				selected_1_way = []
				selected_2_way = {}
				for attri in query.attributes:
					attri_dict[attri] = i
					i = i + 1
					tmp_attri = attribute(attri, result, info, 0)
					selected_domain.append(tmp_attri.domain)
					selected_1_way.append(single_ps[attri])
				for pair in double_dict.keys():  # pair is tuple!
					if not (set(pair) - set(query.attributes)):
						pair_num = []
						for attri in list(pair):
							pair_num.append(attri_dict[attri])
						selected_2_way[tuple(pair_num)] = double_dict[pair]
				marginal_PRISM[weight] = Synthesizer.Maximum_entropy(selected_1_way, selected_2_way, selected_domain,g)
				answer_Prism = query.range_query(marginal_PRISM[weight])
				ERROR_PRISM += utils.MAE(answer_Prism, query.answer, 1)
		MEAN_ERROR_PRISM[ii] = ERROR_PRISM / n
		print('MAE_PRISM for k=',kk,' :', ERROR_PRISM / n)
		# print('done.')
		print('done.')
	for i in range(len(kk_list)):
		print('MAE_PRISM for kk=', kk_list[i], ' :', MEAN_ERROR_PRISM[i])
	print('marginal:', len(attris), 'epsilon:', epsilon, 'query_volume:', query_volume, 'data_size:', data_size)
