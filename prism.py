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


def my_solution2(attributes, measure, epsilon, noisy, partition,g):     #####noisy=0不扰动，=1扰动
	datas = read_data.merge(attributes)
	# table=datas
	table = random.sample(datas, len(datas) // partition)   #按分组数算出每组个数，随机抽样
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
	g=3 #PRISM中的划分粒度
	# print(type(result))
	print('done.')
	n = 10
	k = 2
	epsilon = 5
	query_volume =0.9
	data_size = 10**4

	print('loading data...')
	# path = "E:\PyWorkSpace\program\dataset/data_adult_30-30-1M.csv"
	path="../dataset/data_ipums_30-30-1M.csv"
	# infopath = "E:\PyWorkSpace\program\dataset/info_30-30.csv"
	infopath = "../dataset/info_30-30.csv"
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
	MEAN_ERROR_PRISM = 0
	# MEAN_ERROR_PRISM_OLH = 0
	MEAN_ERROR_PRISM_CUBE = 0
	MEAN_ERROR_LHIO = 0
	MEAN_ERROR_MAX = 0
	MEAN_ERROR_MIN = 0
	MEAN_ERROR_CALM = 0
	MEAN_ERROR_HDG = 0
	# generate queries and raw answers
	for round in range(ROUNDS):
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
			for data in datas:    #填写二边际真实分布表
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
			kk = int(len(attris) / 2) + 1      #########kk=16???????????
			C = math.factorial(kk) // (math.factorial(2) * math.factorial(kk - 2))
			partition = C + len(attris)
			print('collecting single dimensions for weight:', weight, '...')
			single_ps = {}       ########开始算一维分布
			sub_table = tables_dict[weight]     #数据库中权重为1的每一行
			for attri in attris:
				attributes = []
				attributes.append(attribute(attri, sub_table, info, 0))     ####考虑1改0？ipum好像不需要平移
				# 将1-way数据收集并存入single_ps
				single_ps[attri] = my_solution2(attributes, measure, epsilon, 1, partition,g)       ###是所有属性的g粒度一维分布##mysolution2？？？？？？？
			# attri_dict = {}
			i = 0
			print('selecting valuable dimensions...')    #####DELFT选择值得统计二维分布的属性
			# for key in single_ps.keys():
			#     attri_dict[key] = i
			#     i = i + 1
			value_ps = Synthesizer.value_attribute(single_ps)    ######输出选出的属性
			#print(value_ps)
			pairs = utils.get_pair(value_ps)     ###选出的属性的笛卡尔积（两两组合=150种）
			single_list = single_ps.values()
			print('collecting joint dimensions...')     ####开始计算二维属性的联合分布
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
			print('estimating entire model...')      #####通过基集估算所有分布
			ERROR_PRISM = 0
			for query in query_list:   #query_list一共只有10个查询范围
				attri_dict = {}
				i = 0
				selected_domain = []
				selected_1_way = []
				selected_2_way = {}
				for attri in query.attributes:    ####先挑选出待查询属性的一维分布
					attri_dict[attri] = i
					i = i + 1
					tmp_attri = attribute(attri, result, info, 0)
					selected_domain.append(tmp_attri.domain)
					selected_1_way.append(single_ps[attri])
				for pair in double_dict.keys():  # pair is tuple!   ###查找待查询属性的二维组合是否属于基集
					if not (set(pair) - set(query.attributes)):   ###pair是查询属性的子集则返回，不是则执行下一步
						pair_num = []
						for attri in list(pair):
							pair_num.append(attri_dict[attri])
						selected_2_way[tuple(pair_num)] = double_dict[pair]     ####基集中用于计算的二维分布，没有一样的就是空集
				marginal_PRISM[weight] = Synthesizer.Maximum_entropy(selected_1_way, selected_2_way, selected_domain,g)   ######用相关一维分布和二维分布估计查询
				answer_Prism = query.range_query(marginal_PRISM[weight])         ###########？？？？如何实现
				ERROR_PRISM += utils.MAE(answer_Prism, query.answer, 1)
				print(answer_Prism,query_volume)
				# ERROR_PRISM += utils.RE(answer_Prism, query.answer, 1)
		MEAN_ERROR_PRISM += ERROR_PRISM / n / ROUNDS
		print('MAE_PRISM:', ERROR_PRISM / n)
		# print('done.')

		# print('executing PRISM_OLH...')
		# ERROR_PRISM_OLH = 0
		# tables_dict = weighted_frequency_query.spilt_table(result, mea)
		# marginal_PRISM_OLH = {}
		# for weight in tables_dict.keys():
		# 	# 现有代码默认K为D的一半，有提升空间
		# 	kk = int(len(attris) / 2) + 1
		# 	C = math.factorial(kk) // (math.factorial(2) * math.factorial(kk - 2))
		# 	partition = C + len(attris)
		# 	print('collecting single dimensions for weight:', weight, '...')
		# 	single_ps = {}
		# 	sub_table = tables_dict[weight]
		# 	for attri in attris:
		# 		attributes = []
		# 		attributes.append(attribute(attri, sub_table, info, 1))
		# 		# 将1-way数据收集并存入single_ps
		# 		single_ps[attri] = my_solution3(attributes, measure, epsilon, 1, partition, g)
		# 	# attri_dict = {}
		# 	i = 0
		# 	print('selecting valuable dimensions...')
		# 	# for key in single_ps.keys():
		# 	#     attri_dict[key] = i
		# 	#     i = i + 1
		# 	value_ps = Synthesizer.value_attribute(single_ps)
		# 	# print(value_ps)
		# 	pairs = utils.get_pair(value_ps)
		# 	single_list = single_ps.values()
		# 	print('collecting joint dimensions...')
		# 	double_dict = {}
		# 	# attributes = []
		# 	# print(pairs)
		# 	for pair in pairs:
		# 		attributes = []
		# 		print(pair)
		# 		pair_num = []
		# 		for attri in pair:
		# 			new_attri = attribute(attri, result, info, 1)
		# 			attributes.append(new_attri)
		# 		# pair_num.append(attri_dict[attri])
		# 		# key_num = tuple(pair_num)
		# 		double_ps = my_solution3(attributes, measure, epsilon, 1, partition, g)
		# 		double_dict[pair] = double_ps
		# 	print('estimating entire model...')
		# 	ERROR_PRISM_OLH = 0
		# 	for query in query_list:
		# 		attri_dict = {}
		# 		i = 0
		# 		selected_domain = []
		# 		selected_1_way = []
		# 		selected_2_way = {}
		# 		for attri in query.attributes:
		# 			attri_dict[attri] = i
		# 			i = i + 1
		# 			tmp_attri = attribute(attri, result, info, 1)
		# 			selected_domain.append(tmp_attri.domain)
		# 			selected_1_way.append(single_ps[attri])
		# 		for pair in double_dict.keys():  # pair is tuple!
		# 			if not (set(pair) - set(query.attributes)):
		# 				pair_num = []
		# 				for attri in list(pair):
		# 					pair_num.append(attri_dict[attri])
		# 				selected_2_way[tuple(pair_num)] = double_dict[pair]
		# 		marginal_PRISM_OLH[weight] = Synthesizer.Maximum_entropy(selected_1_way, selected_2_way, selected_domain, g)
		# 		answer_Prism_OLH = query.range_query(marginal_PRISM_OLH[weight])
		# 		ERROR_PRISM_OLH += utils.MAE(answer_Prism_OLH, query.answer, 1)
		# MEAN_ERROR_PRISM_OLH += ERROR_PRISM_OLH / n / ROUNDS
		# print('MAE_PRISM_OLH:', ERROR_PRISM_OLH / n)
		# # print('done.')


		# print('executing PRISM_CUBE...')
		# ERROR_PRISM_CUBE = 0
		# tables_dict = weighted_frequency_query.spilt_table(result, mea)
		# marginal_PRISM_CUBE = {}
		# for weight in tables_dict.keys():
		# 	# 现有代码默认K为D的一半，有提升空间
		# 	kk = int(len(attris) / 2) + 1
		# 	C = math.factorial(kk) // (math.factorial(2) * math.factorial(kk - 2))
		# 	partition = C + len(attris)
		# 	print('collecting single dimensions for weight:', weight, '...')
		# 	single_ps = {}
		# 	sub_table = tables_dict[weight]
		# 	for attri in attris:
		# 		attributes = []
		# 		attributes.append(attribute(attri, sub_table, info, 1))
		# 		# 将1-way数据收集并存入single_ps
		# 		single_ps[attri] = my_solution3(attributes, measure, epsilon, 1, partition, g)
		# 	# attri_dict = {}
		# 	i = 0
		# 	print('selecting valuable dimensions...')
		# 	# for key in single_ps.keys():
		# 	#     attri_dict[key] = i
		# 	#     i = i + 1
		# 	value_ps = Synthesizer.value_attribute(single_ps)
		# 	# print(value_ps)
		# 	pairs = utils.get_pair(value_ps)
		# 	single_list = single_ps.values()
		# 	print('collecting joint dimensions...')
		# 	double_dict = {}
		# 	# attributes = []
		# 	# print(pairs)
		# 	for pair in pairs:
		# 		attributes = []
		# 		print(pair)
		# 		pair_num = []
		# 		for attri in pair:
		# 			new_attri = attribute(attri, result, info, 1)
		# 			attributes.append(new_attri)
		# 		# pair_num.append(attri_dict[attri])
		# 		# key_num = tuple(pair_num)
		# 		double_ps = my_solution3(attributes, measure, epsilon, 1, partition, g)
		# 		double_dict[pair] = double_ps
		# 	print('estimating entire model...')
		# 	ERROR_PRISM_CUBE = 0
		# 	for query in query_list:
		# 		attri_dict = {}
		# 		i = 0
		# 		selected_domain = []
		# 		selected_1_way = []
		# 		selected_2_way = {}
		# 		for attri in query.attributes:
		# 			attri_dict[attri] = i
		# 			i = i + 1
		# 			tmp_attri = attribute(attri, result, info, 1)
		# 			selected_domain.append(tmp_attri.domain)
		# 			selected_1_way.append(single_ps[attri])
		# 		for pair in double_dict.keys():  # pair is tuple!
		# 			if not (set(pair) - set(query.attributes)):
		# 				pair_num = []
		# 				for attri in list(pair):
		# 					pair_num.append(attri_dict[attri])
		# 				selected_2_way[tuple(pair_num)] = double_dict[pair]
		# 		marginal_PRISM_CUBE[weight] = Synthesizer.Maximum_entropy(selected_1_way, selected_2_way, selected_domain, g)
		# 		answer_Prism_CUBE = query.range_query(marginal_PRISM_CUBE[weight])
		# 		ERROR_PRISM_CUBE += utils.MAE(answer_Prism_CUBE, query.answer, 1)
		# MEAN_ERROR_PRISM_CUBE += ERROR_PRISM_CUBE / n / ROUNDS
		# print('MAE_PRISM_CUBE:', ERROR_PRISM_CUBE / n)
		# # print('done.')



		print('executing MIN...')
		ERROR_MIN = 0
		tables_dict = weighted_frequency_query.spilt_table(result, mea)
		marginal_MIN = {}
		for weight in tables_dict.keys():
			print('collecting single dimensions for weight:', weight, '...')
			partition = len(attris)
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
			value_ps = Synthesizer.max_value_attribute(single_ps)
			#print(value_ps)
			pairs = utils.get_pair(value_ps)
			single_list = single_ps.values()
			print('collecting joint dimensions...')
			double_dict = {}
			# attributes = []
			#print(pairs)
			for pair in pairs:
				attributes = []
				#print(pair)
				pair_num = []
				for attri in pair:
					new_attri = attribute(attri, result, info, 0)
					attributes.append(new_attri)
					# pair_num.append(attri_dict[attri])
				# key_num = tuple(pair_num)
				double_ps = my_solution2(attributes, measure, epsilon, 1, partition,g)
				double_dict[pair] = double_ps
			print('estimating entire model...')
			ERROR_MIN = 0
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
				marginal_MIN[weight] = Synthesizer.Maximum_entropy(selected_1_way, selected_2_way, selected_domain,g)
				answer_MIN = query.range_query(marginal_MIN[weight])
				ERROR_MIN += utils.MAE(answer_MIN, query.answer, 1)
		MEAN_ERROR_MIN += ERROR_MIN / n / ROUNDS
		print('MAE_MIN:', ERROR_MIN / n)
		print('MAE_PRISM:', ERROR_PRISM / n)

		print('executing MAX...')
		ERROR_MAX = 0
		tables_dict = weighted_frequency_query.spilt_table(result, mea)
		marginal_MAX = {}
		for weight in tables_dict.keys():
			kk = len(attris)
			C = math.factorial(kk) // (math.factorial(2) * math.factorial(kk - 2))
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
			value_ps = Synthesizer.max_value_attribute(single_ps)
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
			ERROR_MAX = 0
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
				marginal_MAX[weight] = Synthesizer.Maximum_entropy(selected_1_way, selected_2_way, selected_domain,g)
				answer_MAX = query.range_query(marginal_MAX[weight])
				ERROR_MAX += utils.MAE(answer_MAX, query.answer, 1)
		MEAN_ERROR_MAX += ERROR_MAX / n / ROUNDS
		print('MAE_MIN:', ERROR_MIN / n)
		print('MAE_MAX:', ERROR_MAX / n)
		print('MAE_PRISM:', ERROR_PRISM / n)
		print('done.')

		# HDG
		print('executing HDG...')
		marginal_HDG = {}
		for weight in tables_dict.keys():
			attributes = []
			for attri in attris:
				sub_table = tables_dict[weight]
				attributes.append(attribute(attri, sub_table, info, 0))
			C = math.factorial(len(attris)) // (math.factorial(2) * math.factorial(len(attris) - 2))
			partition = C + len(attris)
			hdg = control_group.HDG(attributes, epsilon, partition,g)   ####巨慢
			g1=g
			g2=g*5
			ERROR_HDG = 0
			print('answering queries...')
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
					selected_1_way.append(hdg.expend_OneD(attri))  # domain
				for pair in hdg.TwoD.keys():  # pair is tuple!
					if not (set(pair) - set(query.attributes)):
						pair_num = []
						for attri in list(pair):
							pair_num.append(attri_dict[attri])
						selected_2_way[tuple(pair_num)] = hdg.expend_TwoD(pair, domain[0])
				marginal_HDG[weight] = Synthesizer.Maximum_entropy_HDG(selected_1_way, selected_2_way, selected_domain,g1)
				answer_HDG = query.range_query(marginal_HDG[weight])
				ERROR_HDG += utils.MAE(answer_HDG, query.answer, 1)
				# ERROR_HDG += utils.RE(answer_HDG, query.answer, 1)
		MEAN_ERROR_HDG += ERROR_HDG / n / ROUNDS
		print('MAE_HDG:', ERROR_HDG / n)
		print('MAE_MIN:', ERROR_MIN / n)
		print('MAE_MAX:', ERROR_MAX / n)
		print('MAE_PRISM:', ERROR_PRISM / n)
		print('done.')

		#LHIO
		# print('executing LHIO...')
		# marginal_LHIO={}
		# for weight in tables_dict.keys():
		# 	lists = tables_dict[weight]
		# 	lhio = control_group.LHIO(attributes, domain, epsilon,g)
		# 	lhio.collection()
		# 	ERROR_LHIO = 0
		# 	for query in query_list:
		# 		attri_dict = {}
		# 		i = 0
		# 		selected_domain = []
		# 		for attri in query.attributes:
		# 			attri_dict[attri] = i
		# 			i = i + 1
		# 			tmp_attri = attribute(attri, result, info, 1)
		# 			selected_domain.append(tmp_attri.domain)
		# 		involved_view = {}
		# 		for view in lhio.views:  # view is a pair of class
		# 			pair = (view[0].name, view[1].name)
		# 			if not (set(pair) - set(query.attributes)):
		# 				pair_num = []
		# 				for attri in list(pair):
		# 					pair_num.append(attri_dict[attri])
		# 				involved_view[tuple(pair_num)] = view
		# 		hio = control_group.HI(selected_domain, epsilon,g)
		# 		marginal_LHIO[weight] = hio.estimation(lhio, involved_view)
		# 		print(marginal_LHIO[weight])
		# 		if marginal_LHIO[weight]==0:
		# 			answer_LHIO=0
		# 		else:
		# 			answer_LHIO = query.range_query(marginal_LHIO[weight])
		# 		ERROR_LHIO += utils.MAE(answer_LHIO, query.answer, 1)
		# print('MAE_LHIO:', ERROR_LHIO / n)
		print('MAE_HDG:', ERROR_HDG / n)
		print('MAE_MIN:', ERROR_MIN / n)
		print('MAE_MAX:', ERROR_MAX / n)
		print('MAE_PRISM:', ERROR_PRISM / n)

		print('done.')
		#MG
		print('executing CALM...')
		attributes = []
		marginal_MG = {}
		for attri in attris:
			new_attri = attribute(attri, result2, info, 0)
			attributes.append(new_attri)
		for weight in tables_dict.keys():
			lists = tables_dict[weight]
			for attri in attris:
				new_attri = attribute(attri, lists, info, 0)
				attributes.append(new_attri)
			datas = read_data.merge(attributes)
			calm = control_group.CALM(attributes, domain, epsilon,g)
			# print(datas[0])
			print('collecting data...')
			view_data = calm.collection(datas)
			ERROR_MG = 0
			for query in query_list:
				attri_dict = {}
				i = 0
				selected_domain = []
				selected_2_way = {}
				for attri in query.attributes:
					attri_dict[attri] = i
					i = i + 1
					tmp_attri = attribute(attri, result, info, 0)
					selected_domain.append(tmp_attri.domain)
				for pair in calm.attri_name:  # pair is tuple! attri is not class
					if not (set(pair) - set(query.attributes)):
						print(pair)
						pair_num = []
						# print(type(pair))
						for attri in list(pair):
							pair_num.append(attri_dict[attri])
						selected_2_way[tuple(pair_num)] = view_data[pair]
				marginal_MG[weight] = calm.Max_entropy(selected_2_way, selected_domain)
				answer_MG = query.range_query(marginal_MG[weight])
				ERROR_MG += utils.MAE(answer_MG, query.answer, 1)
		MEAN_ERROR_CALM += ERROR_MG / n / ROUNDS
		rounds=round+1
		print('the MAE for round:',round)
		# print('MAE_LHIO:', MEAN_ERROR_LHIO*ROUNDS/rounds)
		print('MAE_MG:', MEAN_ERROR_CALM*ROUNDS/rounds)
		print('MAE_HDG:', MEAN_ERROR_HDG*ROUNDS/rounds)
		print('MAE_MIN:', MEAN_ERROR_MIN*ROUNDS/rounds)
		print('MAE_MAX:', MEAN_ERROR_MAX*ROUNDS/rounds)
		# print('MAE_PRISM_OLH:', MEAN_ERROR_PRISM_OLH * ROUNDS / rounds)
		print('MAE_PRISM:', MEAN_ERROR_PRISM*ROUNDS/rounds)
		print('done.')
	# print('MAE_LHIO:', MEAN_ERROR_LHIO)
	print('MAE_MG:', MEAN_ERROR_CALM)
	print('MAE_HDG:', MEAN_ERROR_HDG)
	print('MAE_MIN:', MEAN_ERROR_MIN)
	print('MAE_MAX:', MEAN_ERROR_MAX)
	# print('MAE_PRISM_OLH:', MEAN_ERROR_PRISM_OLH)
	print('MAE_PRISM:', MEAN_ERROR_PRISM)
	print('marginal:', len(attris), 'epsilon:', epsilon, 'query_volume:', query_volume, 'data_size:', data_size)
