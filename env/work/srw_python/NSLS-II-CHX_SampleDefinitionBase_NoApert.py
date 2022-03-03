#!/usr/bin/env python
import os
from srwl_uti_smp import srwl_opt_setup_transm_from_obj3d
try:
    __IPYTHON__
    import sys
    del sys.argv[1:]
except:
    pass

import srwl_bl
import srwlib
import srwlpy
import math
import srwl_uti_smp_rnd_obj3d
from multiprocessing import Pool

def set_optics(v, names=None, want_final_propagation=True):
    el = []
    pp = []
    if not names:
        names = ['Aperture', 'Aperture_Sample', 'Sample', 'Sample_Detector', 'Detector']
    for el_name in names:
        if el_name == 'Aperture':
            # Aperture: aperture 0.0m
            el.append(srwlib.SRWLOptA(
                _shape=v.op_Aperture_shape,
                _ap_or_ob='a',
                _Dx=v.op_Aperture_Dx,
                _Dy=v.op_Aperture_Dy,
                _x=v.op_Aperture_x,
                _y=v.op_Aperture_y,
            ))
            pp.append(v.op_Aperture_pp)
        elif el_name == 'Aperture_Sample':
            # Aperture_Sample: drift 0.0m
            el.append(srwlib.SRWLOptD(
                _L=v.op_Aperture_Sample_L,
            ))
            pp.append(v.op_Aperture_Sample_pp)
        elif el_name == 'Sample':
            # Sample: sample 0.0m
            el.append(srwl_opt_setup_transm_from_obj3d(
                shape_defs=v.op_Sample_Objects,
                delta = v.op_Sample_delta, atten_len=v.op_Sample_atten_len,
                rx=v.op_Sample_rx, ry=v.op_Sample_ry,
                nx=v.op_Sample_nx, ny=v.op_Sample_ny,
                xc=v.op_Sample_xc, yc=v.op_Sample_yc,
                extTr=v.op_Sample_extTransm))
            pp.append(v.op_Sample_pp)
        elif el_name == 'Sample_Detector':
            # Sample_Detector: drift 0.0m
            el.append(srwlib.SRWLOptD(
                _L=v.op_Sample_Detector_L,
            ))
            pp.append(v.op_Sample_Detector_pp)
        elif el_name == 'Detector':
            # Detector: watch 10.5m
            pass
    if want_final_propagation:
        pp.append(v.op_fin_pp)

    return srwlib.SRWLOptC(el, pp)



varParam = [
    ['name', 's', 'NSLS-II CHX_SampleDefinitionBase', 'simulation name'],

#---Data Folder
    ['fdir', 's', 'sample_def', 'folder (directory) name for reading-in input and saving output data files'],

#---Electron Beam
    ['ebm_nm', 's', '', 'standard electron beam name'],
    ['ebm_nms', 's', '', 'standard electron beam name suffix: e.g. can be Day1, Final'],
    ['ebm_i', 'f', 0.5, 'electron beam current [A]'],
    ['ebm_e', 'f', 3.0, 'electron beam avarage energy [GeV]'],
    ['ebm_de', 'f', 0.0, 'electron beam average energy deviation [GeV]'],
    ['ebm_x', 'f', 0.0, 'electron beam initial average horizontal position [m]'],
    ['ebm_y', 'f', 0.0, 'electron beam initial average vertical position [m]'],
    ['ebm_xp', 'f', 0.0, 'electron beam initial average horizontal angle [rad]'],
    ['ebm_yp', 'f', 0.0, 'electron beam initial average vertical angle [rad]'],
    ['ebm_z', 'f', 0., 'electron beam initial average longitudinal position [m]'],
    ['ebm_dr', 'f', -1.54, 'electron beam longitudinal drift [m] to be performed before a required calculation'],
    ['ebm_ens', 'f', 0.00089, 'electron beam relative energy spread'],
    ['ebm_emx', 'f', 9e-10, 'electron beam horizontal emittance [m]'],
    ['ebm_emy', 'f', 8e-12, 'electron beam vertical emittance [m]'],
    # Definition of the beam through Twiss:
    ['ebm_betax', 'f', 2.02, 'horizontal beta-function [m]'],
    ['ebm_betay', 'f', 1.06, 'vertical beta-function [m]'],
    ['ebm_alphax', 'f', 0.0, 'horizontal alpha-function [rad]'],
    ['ebm_alphay', 'f', 0.0, 'vertical alpha-function [rad]'],
    ['ebm_etax', 'f', 0.0, 'horizontal dispersion function [m]'],
    ['ebm_etay', 'f', 0.0, 'vertical dispersion function [m]'],
    ['ebm_etaxp', 'f', 0.0, 'horizontal dispersion function derivative [rad]'],
    ['ebm_etayp', 'f', 0.0, 'vertical dispersion function derivative [rad]'],

#---Undulator
    ['und_bx', 'f', 0.0, 'undulator horizontal peak magnetic field [T]'],
    ['und_by', 'f', 0.88770981, 'undulator vertical peak magnetic field [T]'],
    ['und_phx', 'f', 0.0, 'initial phase of the horizontal magnetic field [rad]'],
    ['und_phy', 'f', 0.0, 'initial phase of the vertical magnetic field [rad]'],
    ['und_b2e', '', '', 'estimate undulator fundamental photon energy (in [eV]) for the amplitude of sinusoidal magnetic field defined by und_b or und_bx, und_by', 'store_true'],
    ['und_e2b', '', '', 'estimate undulator field amplitude (in [T]) for the photon energy defined by w_e', 'store_true'],
    ['und_per', 'f', 0.02, 'undulator period [m]'],
    ['und_len', 'f', 3.0, 'undulator length [m]'],
    ['und_zc', 'f', 0.0, 'undulator center longitudinal position [m]'],
    ['und_sx', 'i', 1, 'undulator horizontal magnetic field symmetry vs longitudinal position'],
    ['und_sy', 'i', -1, 'undulator vertical magnetic field symmetry vs longitudinal position'],
    ['und_g', 'f', 6.72, 'undulator gap [mm] (assumes availability of magnetic measurement or simulation data)'],
    ['und_ph', 'f', 0.0, 'shift of magnet arrays [mm] for which the field should be set up'],
    ['und_mdir', 's', '', 'name of magnetic measurements sub-folder'],
    ['und_mfs', 's', '', 'name of magnetic measurements for different gaps summary file'],



#---Calculation Types
    #Single-Electron Wavefront Propagation
    ['ws', '', '', 'calculate single-electron (/ fully coherent) wavefront propagation', 'store_true'],
    #Multi-Electron (partially-coherent) Wavefront Propagation
    ['wm', '', '1', 'calculate multi-electron (/ partially coherent) wavefront propagation', 'store_true'],

    ['w_e', 'f', 8000.0, 'photon energy [eV] for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_ef', 'f', -1.0, 'final photon energy [eV] for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_ne', 'i', 1, 'number of points vs photon energy for calculation of intensity distribution'],
    ['w_x', 'f', 0.0, 'central horizontal position [m] for calculation of intensity distribution'],
    ['w_rx', 'f', 1.6e-04, 'range of horizontal position [m] for calculation of intensity distribution'],
    ['w_nx', 'i', 100, 'number of points vs horizontal position for calculation of intensity distribution'],
    ['w_y', 'f', 0.0, 'central vertical position [m] for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_ry', 'f', 1.6e-04, 'range of vertical position [m] for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_ny', 'i', 100, 'number of points vs vertical position for calculation of intensity distribution'],
    ['w_smpf', 'f', 1.0, 'sampling factor for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_meth', 'i', 1, 'method to use for calculation of intensity distribution vs horizontal and vertical position: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"'],
    ['w_prec', 'f', 0.01, 'relative precision for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_u', 'i', 0, 'electric field units: 0- arbitrary, 1- sqrt(Phot/s/0.1%bw/mm^2), 2- sqrt(J/eV/mm^2) or sqrt(W/mm^2), depending on representation (freq. or time)'],
    ['si_pol', 'i', 0, 'polarization component to extract after calculation of intensity distribution: 0- Linear Horizontal, 1- Linear Vertical, 2- Linear 45 degrees, 3- Linear 135 degrees, 4- Circular Right, 5- Circular Left, 6- Total'],
    ['si_type', 'i', 1, 'type of a characteristic to be extracted after calculation of intensity distribution: 0- Single-Electron Intensity, 1- Multi-Electron Intensity, 2- Single-Electron Flux, 3- Multi-Electron Flux, 4- Single-Electron Radiation Phase, 5- Re(E): Real part of Single-Electron Electric Field, 6- Im(E): Imaginary part of Single-Electron Electric Field, 7- Single-Electron Intensity, integrated over Time or Photon Energy'],
    ['w_mag', 'i', 1, 'magnetic field to be used for calculation of intensity distribution vs horizontal and vertical position: 1- approximate, 2- accurate'],

    ['si_fn', 's', 'res_int_se.dat', 'file name for saving calculated single-e intensity distribution (without wavefront propagation through a beamline) vs horizontal and vertical position'],
    ['si_pl', 's', '', 'plot the input intensity distributions in graph(s): ""- dont plot, "x"- vs horizontal position, "y"- vs vertical position, "xy"- vs horizontal and vertical position'],
    ['ws_fni', 's', 'res_int_pr_se.dat', 'file name for saving propagated single-e intensity distribution vs horizontal and vertical position'],
    ['ws_pl', 's', 'xy', 'plot the resulting intensity distributions in graph(s): ""- dont plot, "x"- vs horizontal position, "y"- vs vertical position, "xy"- vs horizontal and vertical position'],

    ['wm_nm', 'i', 10, 'number of macro-electrons (coherent wavefronts) for calculation of multi-electron wavefront propagation'],
    ['wm_na', 'i', 10, 'number of macro-electrons (coherent wavefronts) to average on each node for parallel (MPI-based) calculation of multi-electron wavefront propagation'],
    ['wm_ns', 'i', 10, 'saving periodicity (in terms of macro-electrons / coherent wavefronts) for intermediate intensity at multi-electron wavefront propagation calculation'],
    ['wm_ch', 'i', 0, 'type of a characteristic to be extracted after calculation of multi-electron wavefront propagation: #0- intensity (s0); 1- four Stokes components; 2- mutual intensity cut vs x; 3- mutual intensity cut vs y; 40- intensity(s0), mutual intensity cuts and degree of coherence vs X & Y'],
    ['wm_ap', 'i', 0, 'switch specifying representation of the resulting Stokes parameters: coordinate (0) or angular (1)'],
    ['wm_x0', 'f', 0.0, 'horizontal center position for mutual intensity cut calculation'],
    ['wm_y0', 'f', 0.0, 'vertical center position for mutual intensity cut calculation'],
    ['wm_ei', 'i', 0, 'integration over photon energy is required (1) or not (0); if the integration is required, the limits are taken from w_e, w_ef'],
    ['wm_rm', 'i', 1, 'method for generation of pseudo-random numbers for e-beam phase-space integration: 1- standard pseudo-random number generator, 2- Halton sequences, 3- LPtau sequences (to be implemented)'],
    ['wm_am', 'i', 0, 'multi-electron integration approximation method: 0- no approximation (use the standard 5D integration method), 1- integrate numerically only over e-beam energy spread and use convolution to treat transverse emittance'],
    ['wm_fni', 's', 'res_int_pr_me.dat', 'file name for saving propagated multi-e intensity distribution vs horizontal and vertical position'],
    ['wm_ff', 's', 'ascii', 'format of file name for saving propagated multi-e intensity distribution vs horizontal and vertical position (ascii and hdf5 supported)'],

    ['wm_nmm', 'i', 4, 'number of MPI masters to use'],
    ['wm_ncm', 'i', 10, 'number of Coherent Modes to calculate'],
    ['wm_acm', 's', 'SP', 'coherent mode decomposition algorithm to be used (supported algorithms are: "SP" for SciPy, "SPS" for SciPy Sparse, "PM" for Primme, based on names of software packages)'],
    ['wm_nop', '', '', 'switch forcing to do calculations ignoring any optics defined (by set_optics function)', 'store_true'],

    ['wm_fnmi', 's', '', 'file name of input cross-spectral density / mutual intensity; if this file name is supplied, the initial cross-spectral density (for such operations as coherent mode decomposition) will not be calculated, but rathre it will be taken from that file.'],
    ['wm_fncm', 's', 'chx_res_pr_dir_100k_cm.h5', 'file name of input coherent modes; if this file name is supplied, the eventual partially-coherent radiation propagation simulation will be done based on propagation of the coherent modes from that file.'],

    ['wm_fbk', '', '', 'create backup file(s) with propagated multi-e intensity distribution vs horizontal and vertical position and other radiation characteristics', 'store_true'],

    # Optics parameters
    ['op_r', 'f', 20.5, 'longitudinal position of the first optical element [m]'],
    # Former appParam:
    ['rs_type', 's', 'u', 'source type, (u) idealized undulator, (t), tabulated undulator, (m) multipole, (g) gaussian beam'],

#---Beamline optics:
    # Aperture: aperture
    ['op_Aperture_shape', 's', 'r', 'shape'],
    #['op_Aperture_Dx', 'f', 3e-05, 'horizontalSize'],
    ['op_Aperture_Dx', 'f', 3e-05, 'horizontalSize'],
    #['op_Aperture_Dy', 'f', 1.67e-05, 'verticalSize'],
    ['op_Aperture_Dy', 'f', 3e-04, 'verticalSize'],
    #['op_Aperture_Dy', 'f', 3e-01, 'verticalSize'],
    ['op_Aperture_x', 'f', 0.0, 'horizontalOffset'],
    ['op_Aperture_y', 'f', 0.0, 'verticalOffset'],

    # Aperture_Sample: drift
    ['op_Aperture_Sample_L', 'f', 0.5, 'length'],

    # Obstacle: obstacle
    ['op_Obstacle_shape', 's', 'c', 'shape'],
    ['op_Obstacle_Dx', 'f', 1e-05, 'horizontalSize'],
    ['op_Obstacle_Dy', 'f', 1e-05, 'verticalSize'],
    ['op_Obstacle_x', 'f', 0.0, 'horizontalOffset'],
    ['op_Obstacle_y', 'f', 0.0, 'verticalOffset'],

    # Sample: sample
    ['op_Sample_file_path', 's', 'sample.tif', 'imageFile'],
    ['op_Sample_outputImageFormat', 's', 'tif', 'outputImageFormat'],
    ['op_Sample_position', 'f', 0.0, 'position'],
    ['op_Sample_delta', 'f', 4.773e-05, 'refractiveIndex'],
    ['op_Sample_atten_len', 'f', 2.48644e-06, 'attenuationLength'],
    ['op_Sample_xc', 'f', 0.0, 'horizontalCenterCoordinate'],
    ['op_Sample_yc', 'f', 0.0, 'verticalCenterCoordinate'],
    ['op_Sample_zc', 'f', 0.0, 'depthCenterCoordinate'],
    ['op_Sample_rx', 'f', 20.e-06, 'rx'],
    ['op_Sample_ry', 'f', 20.e-06, 'ry'],
    ['op_Sample_rz', 'f', 20.e-06, 'rz'],
    ['op_Sample_obj_size_min', 'f', 25.e-09, 'obj_size_min'],
    ['op_Sample_obj_size_max', 'f', 250.e-09, 'obj_size_max'],
    ['op_Sample_extTransm', 'i', 1, 'transmissionImage'],
    ['op_Sample_nx', 'i', 2000, 'nx'],
    ['op_Sample_ny', 'i', 2000, 'ny'],

    # Sample_Detector: drift
    ['op_Sample_Detector_L', 'f', 10.0, 'length'],

#---Propagation parameters
    ['op_Aperture_pp', 'f',        [0, 0, 1.0, 0, 0, 1.0, 55.0, 1.0, 55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Aperture'],
    ['op_Aperture_Sample_pp', 'f', [0, 0, 1.0, 3, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Aperture_Sample'],
    ['op_Sample_pp', 'f',          [0, 0, 1.0, 0, 0, 1.0, 165.0, 1.0, 165.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Sample'],
    ['op_Sample_Detector_pp', 'f', [0, 0, 1.0, 3, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Sample_Detector'],
    ['op_fin_pp', 'f',             [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'final post-propagation (resize) parameters'],

    #[ 0]: Auto-Resize (1) or not (0) Before propagation
    #[ 1]: Auto-Resize (1) or not (0) After propagation
    #[ 2]: Relative Precision for propagation with Auto-Resizing (1. is nominal)
    #[ 3]: Allow (1) or not (0) for semi-analytical treatment of the quadratic (leading) phase terms at the propagation
    #[ 4]: Do any Resizing on Fourier side, using FFT, (1) or not (0)
    #[ 5]: Horizontal Range modification factor at Resizing (1. means no modification)
    #[ 6]: Horizontal Resolution modification factor at Resizing
    #[ 7]: Vertical Range modification factor at Resizing
    #[ 8]: Vertical Resolution modification factor at Resizing
    #[ 9]: Type of wavefront Shift before Resizing (not yet implemented)
    #[10]: New Horizontal wavefront Center position after Shift (not yet implemented)
    #[11]: New Vertical wavefront Center position after Shift (not yet implemented)
    #[12]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Horizontal Coordinate
    #[13]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Vertical Coordinate
    #[14]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Longitudinal Coordinate
    #[15]: Optional: Orientation of the Horizontal Base vector of the Output Frame in the Incident Beam Frame: Horizontal Coordinate
    #[16]: Optional: Orientation of the Horizontal Base vector of the Output Frame in the Incident Beam Frame: Vertical Coordinate
]



def epilogue():
    pass


def main(idx, per_proc = 5):
    v = srwl_bl.srwl_uti_parse_options(srwl_bl.srwl_uti_ext_options(varParam), use_sys_argv=True)
    
    listObjInit = srwl_uti_smp_rnd_obj3d.setup_list_obj3d(
        _n = 100, #Number of 3D nano-objects
        _ranges = [0.95*v.op_Sample_rx, 0.95*v.op_Sample_ry, v.op_Sample_rz], #Ranges of horizontal, vertical and longitudinal position within which the 3D objects are defined
        #_ranges = [rx, ry, rz], #Ranges of horizontal, vertical and longitudinal position within which the 3D objects are defined
        _cen = [v.op_Sample_xc, v.op_Sample_yc, v.op_Sample_zc], #Horizontal, Vertical and Longitudinal coordinates of center position around which the 3D objects are defined
        _dist = 'uniform', #Type (and eventual parameters) of distributions of 3D objects
        _obj_shape = ['S', 'uniform', v.op_Sample_obj_size_min, v.op_Sample_obj_size_max], #Type of 3D objects, their distribution type and parameters (min. and max. diameter for the 'uniform' distribution)
        _allow_overlap = False, #Allow or not the 3D objects to overlap
        _fp = os.path.join(os.getcwd(), 'sample_def', v.op_Sample_file_path),
        _seed = 0,
    )

    timeStep = 0.1 #Time step between different Sample "snapshots" / scattering patterns
    timeInterv = 20 #Total time interval covered by the "snapshots"
    listObjBrownian = srwl_uti_smp_rnd_obj3d.brownian_motion3d(
        _obj_crd = listObjInit, #Initial list of 3D objects
        _viscosity = 1.e-3, #[Pa*s]
        _temperature = 293, #[K]
        _timestep = timeStep, #[s]
        _duration = timeInterv, #[s]
        _seed = 0,
        _fp = os.path.join(os.getcwd(), 'sample_def', 'bm_%d.def'))
        
    #names = ['Aperture','Aperture_Sample','Sample','Sample_Detector','Detector']
    names = ['Sample','Sample_Detector','Detector']
    #names = ['Aperture','Sample_Detector','Detector']
    for i in range(idx * per_proc, (idx + 1) * per_proc):
        if i >= len(listObjBrownian): break
        v.op_Sample_Objects = listObjBrownian[i]
        v.wm_fni = 'res_int_pr_me_noapert_%d.dat' % i
        op = set_optics(v, names, True)
        srwl_bl.SRWLBeamline(_name=v.name).calc_all(v, op)

if __name__ == '__main__':
    main(0, 1)
    #with Pool(20) as p:
    #    p.map(main, range(40))
    epilogue()