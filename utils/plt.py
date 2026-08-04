from utils.utils import load
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import mplhep as hep
plt.style.use(hep.style.CMS)

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--ckpts', default='ckpts',
                    help="Name of the ckpts folder")


args = parser.parse_args()
#a=load(args.ckpts + '/' +args.restore_file+ '.resolutions')
a=load(args.ckpts + '/best.resolutions')
colors = {
#    'pfMET': 'black',
#    'puppiMET': 'red',
#    'deepMETResponse': 'blue',
#    'deepMETResolution': 'green',
    'MET':  'magenta',
}
label_arr = {
    'MET':     'Graph MET' ,
#    'pfMET':    'PF MET',
#    'puppiMET': 'PUPPI MET',
#    'deepMETResponse': 'DeepMETResponse',
#    'deepMETResolution': 'DeepMETResolution',
}
resolutions_arr = {
    'MET':      [[],[],[]],
#    'pfMET':    [[],[],[]],
#    'puppiMET': [[],[],[]],
#    'deepMETResponse': [[],[],[]],
#    'deepMETResolution': [[],[],[]],
}
# a saved entry is a np.histogram tuple (values, bin_edges); plot values vs bin centers
def centers(hist):
    edges = np.asarray(hist[1])
    return 0.5 * (edges[:-1] + edges[1:]), np.asarray(hist[0])

for key in resolutions_arr:
         plt.figure(1)
         xx, yy = centers(a[key]['u_perp_resolution'])
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])
         plt.figure(2)
         xx, yy = centers(a[key]['u_perp_scaled_resolution'])
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])
         plt.figure(3)
         xx, yy = centers(a[key]['u_par_resolution'])
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])
         plt.figure(4)
         xx, yy = centers(a[key]['u_par_scaled_resolution'])
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])
         plt.figure(5)
         xx, yy = centers(a[key]['R'])
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])

if(True):
    model_dir=args.ckpts+'/'
    #model_dir=args.ckpts+'/'+args.restore_file+'_'
    plt.figure(1)
    plt.axis([0, 400, 0, 35])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'$\sigma (u_{\perp})$ [GeV]')
    plt.legend()
    plt.savefig(model_dir+'resol_perp.png')
    plt.clf()
    plt.close()

    plt.figure(2)
    plt.axis([0, 400, 0, 35])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Scaled $\sigma (u_{\perp})$ [GeV]')
    plt.legend()
    plt.savefig(model_dir+'resol_perp_scaled.png')
    plt.clf()
    plt.close()

    plt.figure(3)
    plt.axis([0, 400, 0, 60])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'$\sigma (u_{\parallel})$ [GeV]')
    plt.legend()
    plt.savefig(model_dir+'resol_parallel.png')
    plt.clf()
    plt.close()

    plt.figure(4)
    plt.axis([0, 400, 0, 60])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Scaled $\sigma (u_{\parallel})$ [GeV]')
    plt.legend()
    plt.savefig(model_dir+'resol_parallel_scaled.png')
    plt.clf()
    plt.close()

    plt.figure(5)
    plt.axis([0, 400, 0, 1.2])
    plt.axhline(y=1.0, color='black', linestyle='-.')
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Response $-\frac{<u_{\parallel}>}{<q_{T}>}$')
    plt.legend()
    plt.savefig(model_dir+'response_parallel.png')
    plt.clf()
    plt.close()



