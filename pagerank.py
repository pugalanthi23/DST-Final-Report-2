import csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight='weight', dangling=None):
	if len(G) == 0:
		return {}

	if not G.is_directed():
		D = G.to_directed()
	else:
		D = G

	W = nx.stochastic_graph(D, weight=weight)
	N = W.number_of_nodes()

	if nstart is None:
		x = dict.fromkeys(W, 1.0 / N)
	else:
		s = float(sum(nstart.values()))
		x = dict((k, v / s) for k, v in nstart.items())

	if personalization is None:
		p = dict.fromkeys(W, 1.0 / N)
	else:
		s = float(sum(personalization.values()))
		p = dict((k, v / s) for k, v in personalization.items())

	if dangling is None:
		dangling_weights = p
	else:
		s = float(sum(dangling.values()))
		dangling_weights = dict((k, v/s) for k, v in dangling.items())
	dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

	for _ in range(max_iter):
		xlast = x
		x = dict.fromkeys(xlast.keys(), 0)
		danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
		for n in x:
			for nbr in W[n]:
				x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
			x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]

		err = sum([abs(x[n] - xlast[n]) for n in x])
		if err < N*tol:
			return x

def find_influencer_user(rank_list , cluster_details):
	new_dict = []
	for user, rank in list(rank_list.items())[2:]:
		if str(user) in cluster_details:
			new_dict.append((cluster_details[str(user)], rank, user))
	return sorted(new_dict)

file = pd.read_csv("location_cluster.csv")

G = nx.Graph()

for user, follower in zip(file['userid'], file['followers']) :
    G.add_node(user)
    G.add_edge(user, follower)

ls = pagerank(G, 0.4)
#print(len(ls))

cluster = pd.read_csv('location_cluster.csv')
l = {str(x):y for x, y in zip(cluster['userid'], cluster['cluster'])}
#print(l)
out = find_influencer_user(ls, l)
#print(out)
c = 0
print("User_Id    Rank    Cluster")
print('-'*27)
for clus, rank, uid in out:
	if clus == c:
		print("{} {}  {}".format(str(uid).ljust(9, ' '), rank, clus))
		c += 1