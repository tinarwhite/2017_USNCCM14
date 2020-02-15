# 2017_USNCCM14
A Spatial Clustering Algorithm for Constructing Local Reduced order Bases for Nonlinear Model Reduction

Included is solv_uns.py, which can be used to generate a more unsteady inviscid Burger’s problems. Included is also solv_rom_hrom.py which implements hyper reduction. Included is solv_rom_col_row_sparse_energy_content.py, which runs a ROM with a sparse implementation of both row and column clustering. This setup chooses the number of singular vectors to take using the energy content criteria, while solv_rom_col_row_sparse.py takes a constant number of vectors you've defined. Presented at USNCCM14 in July 2017. 

To run the model reduction code, you would need to request the pymortestbed python implementation of reduced order modeling built by Matt Zahr. 

# 2016_CS229
A Clustering Algorithm for Reduced Order Modeling of Shock Waves

The rom_stuff.py files contains the local implementation of a point selection / mesh coarsening method from my AA290 technical report. 

More detailed information including reports and presentations at tinarwhite.com

