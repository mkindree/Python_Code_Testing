import numpy as np
import grid_conversion as gc

def ConstructionFromPOD(num_snaps, POD_modes, t_ind_min, Phi_uf_g, Phi_vf_g, Phi_wf_g, a, I, J):

    Phi_uf = gc.fromgrid(Phi_uf_g, J, I)
    Phi_vf = gc.fromgrid(Phi_vf_g, J, I)
    Phi_wf = gc.fromgrid(Phi_wf_g, J, I)
    
    num_POD_modes = np.size(POD_modes)
#    num_POD_modes = POD_modes.shape(axis=0)
    num_vect = np.size(Phi_uf, axis=0)
    
    
    # Reconstruct POD modes
#    uf_const = np.zeros((num_vect, num_snaps, num_POD_modes))
#    vf_const = np.zeros((num_vect, num_snaps, num_POD_modes))
#    wf_const = np.zeros((num_vect, num_snaps, num_POD_modes))
    if num_POD_modes > 0:
#        for mode in POD_modes:
#            mode_ind = np.where(mode == np.array(POD_modes))
#            print mode
#            print np.size(Phi_uf[:, mode])
#            print np.size(a[t_ind_min : t_ind_min + num_snaps, mode].T)
#            uf_const[:, :, mode_ind] = Phi_uf[:, mode]*a[t_ind_min : t_ind_min + num_snaps, mode].T
#            vf_const[:, :, mode_ind] = Phi_vf[:, mode]*a[t_ind_min : t_ind_min + num_snaps, mode].T
#            wf_const[:, :, mode_ind] = Phi_wf[:, mode]*a[t_ind_min : t_ind_min + num_snaps, mode].T
        uf_const = Phi_uf[:, POD_modes].dot(a[t_ind_min : t_ind_min + num_snaps, POD_modes].T)
        vf_const = Phi_vf[:, POD_modes].dot(a[t_ind_min : t_ind_min + num_snaps, POD_modes].T)
        wf_const = Phi_wf[:, POD_modes].dot(a[t_ind_min : t_ind_min + num_snaps, POD_modes].T)
    else:
        uf_const = np.zeros((num_vect, num_snaps))
        vf_const = np.zeros((num_vect, num_snaps))
        wf_const = np.zeros((num_vect, num_snaps))
#    print uf_const.shape
#    uf_const = np.sum(uf_const, axis=2)
#    vf_const = np.sum(vf_const, axis=2)
#    wf_const = np.sum(wf_const, axis=2)
    
    return uf_const, vf_const, wf_const
