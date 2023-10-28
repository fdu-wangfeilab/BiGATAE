import scanpy as sc

def slice_proprocess(slice):
    sc.pp.filter_genes(slice, min_counts = 15)
    sc.pp.filter_cells(slice, min_counts = 100)
    return slice