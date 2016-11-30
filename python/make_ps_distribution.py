import numpy as np
from scipy.stats import truncnorm as trandn
from prettytable import PrettyTable
import linecache
import random

def string_to_float(seq):
        for x in seq:
            try:
                yield float(x)
            except ValueError:
                yield x

def get_gauss_distribution(output='initial_distribution_', input='sequence_totrack.tfs', sigmas=3,
                            beam_t='LHC', n_part=100, seed=123):

    filename_out = output

    twiss_file = "../input/"+input
    #print twiss_file    
    
    variables = linecache.getline(twiss_file, 46).split()[1:]
    twiss = list(string_to_float(linecache.getline(twiss_file, 48).split()))

    x0 = twiss[variables.index('X')]
    y0 = twiss[variables.index('Y')]
    px0 = twiss[variables.index('PX')]
    py0 = twiss[variables.index('PX')]


    betx = twiss[variables.index('BETX')]
    bety = twiss[variables.index('BETY')]

    alfx = twiss[variables.index('ALFX')]
    alfy = twiss[variables.index('ALFY')]

    dx = twiss[variables.index('DX')]
    dpx = twiss[variables.index('DPX')]
    dy = twiss[variables.index('DY')]
    dpy = twiss[variables.index('DPY')]


    #========================================
    # Seed
    #========================================

    np.random.seed(seed)

    beam_type = beam_t

    if beam_type == 'LHC':
        beam = dict(type='LHC', energy=450., emit_nom=3.5e-6, emit_n=3.5e-6, intensity=288 * 1.15e11,
                    dpp_t=3e-4, n_particles=10000, n_sigma=5)
    elif beam_type == 'LIU':
        beam = dict(type='LIU', energy=450., emit_nom=3.5e-6, emit_n=1.37e-6, intensity=288 * 2.0e11,
                    dpp_t=3e-4, n_particles=1000, n_sigma=5)
    elif beam_type == 'LHC-meas':
        beam = dict(type='LHC-meas', energy=450., emit_nom=3.5e-6, emit_n=2.0e-6, intensity=288 * 1.2e11,
                    dpp_t=3e-4, n_particles=1e6, n_sigma=6.8)
    elif beam_type == 'FT':
        beam = dict(type='FT', energy=400., emit_nom=12e-6, emit_n=8e-6, intensity=3e12,
                    dpp_t=2.5e-4, n_particles=1e6, n_sigma=6.8)
                #dpp_t=4e-4
    else:
        beam = dict(type='HL-LHC', energy=450., emit_nom=3.5e-6, emit_n=2.08e-6, intensity=288 * 2.32e11,
                    dpp_t=3e-4, n_particles=1e6, n_sigma=5)

    n_sigma = sigmas

    emit_n = beam['emit_n']  # [mm.mrad]
    dpp_t = beam['dpp_t']

    n = n_part

    m_p = 0.938

    #========================================
    # Beam parameters
    #========================================

    p = beam['energy']  # momentum in GeV

    gamma = np.sqrt(m_p ** 2 + p ** 2) / m_p
    beta0 = np.sqrt(gamma**2 - 1) / gamma
    beta_r = np.sqrt(1. - 1. / gamma ** 2)
    emit_x = emit_n / (beta_r * gamma)
    emit_y = emit_n / (beta_r * gamma)

    e0 = np.sqrt(p**2 + m_p**2)

    #========================================
    # Delta p / p
    #========================================

    ddp = (np.random.rand(n) * 2. * dpp_t) - dpp_t

    de = ddp * e0 * beta0**2
    pt = de / p

    #==============================================
    # Bivariate normal distributions
    #==============================================

    sx = np.sqrt(emit_x * betx)
    n_x = trandn(-1 * n_sigma, n_sigma, scale=sx).rvs(n)
    x = x0 + n_x + (dx * ddp)
    px = px0 + (trandn(-1 * n_sigma, n_sigma, scale=sx).rvs(n) - alfx * n_x) / betx + (dpx * ddp)

    sy = np.sqrt(emit_y * bety)
    n_y = trandn(-1 * n_sigma, n_sigma, scale=sy).rvs(n)
    y = y0 + n_y + (dy * ddp)
    py = py0 + (trandn(-1 * n_sigma, n_sigma, scale=sy).rvs(n) - alfy * n_y) / bety + (dpy * ddp)

    file_head = '../input/distributionheader.txt'
    head = []
    with open(file_head, 'r') as ff:
        for i in range(6):
            head.append(ff.readline())

    normal = PrettyTable(['*', 'NUMBER', 'TURN', 'X', 'PX', 'Y', 'PY', 'T', 'PT', 'S', 'E'])
    normal.align = 'r'
    normal.border = False
    normal.add_row(['$', '%d', '%d', '%le', '%le', '%le', '%le', '%le', '%le', '%le', '%le'])
    i = 0
    with open(filename_out + '.txt', 'w') as fp:
        for i in range(len(head)):
            fp.write(head[i])
        for i in range(len(x)):
            normal.add_row([' ', i + 1, 0, x[i], px[i], y[i], py[i], 0.0000000, pt[i], 0.0000000, beam['energy']])
        fp.write(normal.get_string())

    return x, px, y, py,pt



def get_halo_distribution(output='initial_distribution_halo_', input='sequence_totrack.tfs', n_halo=5, beam_t='LHC', n_part=100, seed=123):

    filename_out = output

    twiss_file = "../input/"+input

    variables = linecache.getline(twiss_file, 46).split()[1:]
    twiss = list(string_to_float(linecache.getline(twiss_file, 48).split()))

    x0 = twiss[variables.index('X')]
    y0 = twiss[variables.index('Y')]
    px0 = twiss[variables.index('PX')]
    py0 = twiss[variables.index('PX')]


    betx = twiss[variables.index('BETX')]
    bety = twiss[variables.index('BETY')]

    alfx = twiss[variables.index('ALFX')]
    alfy = twiss[variables.index('ALFY')]

    dx = twiss[variables.index('DX')]
    dpx = twiss[variables.index('DPX')]
    dy = twiss[variables.index('DY')]
    dpy = twiss[variables.index('DPY')]


    #========================================
    # Seed
    #========================================

    np.random.seed(seed)

    beam_type = beam_t

    if beam_type == 'LHC':
        beam = dict(type='LHC', energy=450, emit_nom=3.5e-6, emit_n=3.5e-6, intensity=288 * 1.15e11,
                    dpp_t=3e-4, n_particles=10000, n_sigma=5)
    elif beam_type == 'LIU':
        beam = dict(type='LIU', energy=450, emit_nom=3.5e-6, emit_n=1.37e-6, intensity=288 * 2.0e11,
                    dpp_t=3e-4, n_particles=1000, n_sigma=5)
    elif beam_type == 'LHC-meas':
        beam = dict(type='LHC-meas', energy=450, emit_nom=3.5e-6, emit_n=2.0e-6, intensity=288 * 1.2e11,
                    dpp_t=3e-4, n_particles=1e6, n_sigma=6.8)
    else:
        beam = dict(type='HL-LHC', energy=450, emit_nom=3.5e-6, emit_n=2.08e-6, intensity=288 * 2.32e11,
                    dpp_t=3e-4, n_particles=1e6, n_sigma=5)

    emit_n = beam['emit_n']  # [mm.mrad]
    dpp_t = beam['dpp_t']

    n = n_part

    m_p = 0.938

    #========================================
    # Beam parameters
    #========================================

    p = beam['energy']  # momentum in GeV

    gamma = np.sqrt(m_p ** 2 + p ** 2) / m_p
    beta0 = np.sqrt(gamma**2 - 1) / gamma
    beta_r = np.sqrt(1. - 1. / gamma ** 2)
    emit_x = emit_n / (beta_r * gamma)
    emit_y = emit_n / (beta_r * gamma)

    e0 = np.sqrt(p**2 + m_p**2)

    #========================================
    # Delta p / p
    #========================================

    ddp = (np.random.rand(n) * 2. * dpp_t) - dpp_t

    de = ddp * e0 * beta0**2
    pt = de / p
    #==============================================
    # Halo distributions
    #==============================================

    psi = (np.random.rand(n) * 2. * np.pi)

    ddp_t = ddp


    x1 = x0 + n_halo * np.sqrt(betx * emit_x) * np.cos(psi) + (dx * ddp_t)
    px1 = px0 - (n_halo * np.sqrt(emit_x / betx) * (np.sin(psi) + alfx * np.cos(psi))) + (dpx * ddp_t)

    psiy = (np.random.rand(n) * 2 * np.pi)

    y1 = y0 + n_halo * np.sqrt(bety * emit_y) * np.cos(psiy) + (dy * ddp_t)
    py1 = py0 - (n_halo * np.sqrt(emit_y / bety) * (np.sin(psiy) + alfy * np.cos(psiy))) + (dpy * ddp_t)

    file_head = '../input/distributionheader.txt'
    head = []
    with open(file_head, 'r') as ff:
        for i in range(6):
            head.append(ff.readline())

    halo_tab = PrettyTable(['*', 'NUMBER', 'TURN', 'X', 'PX', 'Y', 'PY', 'T', 'PT', 'S', 'E'])
    halo_tab.align = 'r'
    halo_tab.border = False
    halo_tab.add_row(['$', '%d', '%d', '%le', '%le', '%le', '%le', '%le', '%le', '%le', '%le'])


    with open(filename_out + '_halo_' + str(int(n_halo * 10)) + '.txt', 'w') as fp:
        for i in range(len(head)):
            fp.write(head[i])
        for i in range(len(x1)):
            halo_tab.add_row([' ', i + 1, 0, x1[i], px1[i], y1[i], py1[i],
                              0.0000000, pt[i], 0.0000000,  beam['energy']])
        fp.write(halo_tab.get_string())

    return x1, px1, y1, py1,pt


def get_fat_halo(output='initial_distribution_f_halo_', input='sequence_totrack.tfs', n_halo=(4, 5), beam_t='LHC', n_part=100, seed=123):

    filename_out = output

    twiss_file = "../input/"+input

    variables = linecache.getline(twiss_file, 46).split()[1:]
    twiss = list(string_to_float(linecache.getline(twiss_file, 48).split()))

    x0 = twiss[variables.index('X')]
    y0 = twiss[variables.index('Y')]
    px0 = twiss[variables.index('PX')]
    py0 = twiss[variables.index('PX')]


    betx = twiss[variables.index('BETX')]
    bety = twiss[variables.index('BETY')]

    alfx = twiss[variables.index('ALFX')]
    alfy = twiss[variables.index('ALFY')]

    dx = twiss[variables.index('DX')]
    dpx = twiss[variables.index('DPX')]
    dy = twiss[variables.index('DY')]
    dpy = twiss[variables.index('DPY')]


    #========================================
    # Seed
    #========================================

    np.random.seed(seed)
    random.seed(seed)

    beam_type = beam_t

    if beam_type == 'LHC':
        beam = dict(type='LHC', energy=450, emit_nom=3.5e-6, emit_n=3.5e-6, intensity=288 * 1.15e11,
                    dpp_t=3e-4, n_particles=10000, n_sigma=5)
    elif beam_type == 'LIU':
        beam = dict(type='LIU', energy=450, emit_nom=3.5e-6, emit_n=1.37e-6, intensity=288 * 2.0e11,
                    dpp_t=3e-4, n_particles=1000, n_sigma=5)
    elif beam_type == 'LHC-meas':
        beam = dict(type='LHC-meas', energy=450, emit_nom=3.5e-6, emit_n=2.0e-6, intensity=288 * 1.2e11,
                    dpp_t=3e-4, n_particles=1e6, n_sigma=6.8)
    elif beam_type == 'FT':
        beam = dict(type='FT', energy=400., emit_nom=12e-6, emit_n=8e-6, intensity=3e12,
                    dpp_t=4e-4, n_particles=1e6, n_sigma=6.8)
    else:
        beam = dict(type='HL-LHC', energy=450, emit_nom=3.5e-6, emit_n=2.08e-6, intensity=288 * 2.32e11,
                    dpp_t=3e-4, n_particles=1e6, n_sigma=5)

    emit_n = beam['emit_n']  # [mm.mrad]
    dpp_t = beam['dpp_t']

    n = n_part

    m_p = 0.938

    #========================================
    # Beam parameters
    #========================================

    p = beam['energy']  # momentum in GeV

    gamma = np.sqrt(m_p ** 2 + p ** 2) / m_p
    beta0 = np.sqrt(gamma**2 - 1) / gamma
    beta_r = np.sqrt(1. - 1. / gamma ** 2)
    emit_x = emit_n / (beta_r * gamma)
    emit_y = emit_n / (beta_r * gamma)

    e0 = np.sqrt(p**2 + m_p**2)

    #========================================
    # Delta p / p
    #========================================

    ddp = (np.random.rand(n) * 2. * dpp_t) - dpp_t

    de = ddp * e0 * beta0**2
    pt = de / p
    #==============================================
    # Halo distributions
    #==============================================

    psi = (np.random.rand(n) * 2. * np.pi)

    ddp_t = dpp_t

    width = np.array([random.uniform(n_halo[0], n_halo[1]) for i in range(0, n)])

    x1 = x0 + width * np.sqrt(betx * emit_x) * np.cos(psi) + (dx * ddp_t)
    px1 = px0 - (width * np.sqrt(emit_x / betx) * (np.sin(psi) + alfx * np.cos(psi))) + (dpx * ddp_t)

    psiy = (np.random.rand(n) * 2 * np.pi)

    y1 = y0 + width * np.sqrt(bety * emit_y) * np.cos(psiy) + (dy * ddp_t)
    py1 = py0 - (width * np.sqrt(emit_y / bety) * (np.sin(psiy) + alfy * np.cos(psiy))) + (dpy * ddp_t)

    file_head = '../input/distributionheader.txt'
    head = []
    with open(file_head, 'r') as ff:
        for i in range(6):
            head.append(ff.readline())

    halo_tab = PrettyTable(['*', 'NUMBER', 'TURN', 'X', 'PX', 'Y', 'PY', 'T', 'PT', 'S', 'E'])
    halo_tab.align = 'r'
    halo_tab.border = False
    halo_tab.add_row(['$', '%d', '%d', '%le', '%le', '%le', '%le', '%le', '%le', '%le', '%le'])


    with open(filename_out + '_fat_halo_' + str(int(n_halo[1] * 10)) + '.txt', 'w') as fp:
        for i in range(len(head)):
            fp.write(head[i])
        for i in range(len(x1)):
            halo_tab.add_row([' ', i + 1, 0, x1[i], px1[i], y1[i], py1[i],
                              0.0000000, pt[i], 0.0000000,  beam['energy']])
        fp.write(halo_tab.get_string())

    return x1, px1, y1, py1,pt
