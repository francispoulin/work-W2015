import matplotlib.pyplot as plt
import numpy as np

M = range(7,15)

par_spar_s  = [0.081899659999999999, 0.089061419999999974, 0.10120592000000003, 0.12850600000000001, 0.18287281999999996, 0.29140401999999993, 0.49553008000000015, 0.88820650000000034]
par_spar_p2 = [0.23604101999999996, 0.24822545999999998, 0.26299832000000001, 0.29268623999999999, 0.33874894, 0.45001777999999992, 0.68050370000000004, 1.21853224]
par_spar_p4 = [0.34859795999999998, 0.36345010000000011, 0.37267954000000003, 0.40196120000000002, 0.46896671999999995, 0.59354128000000017, 0.84942808000000003, 1.6569714200000001]

par_step_s  = [0.10046695999999995, 0.10384902, 0.11404341999999996, 0.12997647999999998, 0.16325813999999997, 0.22692861999999997, 0.36708930000000012, 0.7317871600000001]
par_step_p2 = [0.89963456000000019, 0.8695112399999998, 0.86011658000000002, 0.85887494000000009, 0.9212118600000001, 0.97367962000000019, 1.1192925399999998, 1.3030841199999998]
par_step_p4 = [1.2910923600000002, 1.2841258399999997, 1.2782660399999997, 1.2907426800000001, 1.3096980400000005, 1.3597024599999998, 1.4460064800000001, 1.6115786199999997]

def eff(t1, tP, p):
    # return np array of values that correspond to the efficiencies of parallel operations
    # efficiency is defined as: E_p = t1 / (p * tP)
    # t1 = list of serial times
    # tP = list of parallel times
    # p  = number of processors
    return np.array(t1) / (p*np.array(tP))

eff_spar_p2 = eff(par_spar_s, par_spar_p2, 2)
eff_spar_p4 = eff(par_spar_s, par_spar_p4, 4)
eff_step_p2 = eff(par_step_s, par_step_p2, 2)
eff_step_p4 = eff(par_step_s, par_step_p4, 4)

def eff_plot(op, title, eff_p2, eff_p4):
    m = M[:len(eff_p2)]
    plt.xlim([5.5, 14.5])
    plt.xlabel('$m$ ($M=2^m$, grid size is $(M+2)\\times(M+2)$ )')
    plt.ylabel('Efficiency')
    plt.title('Efficiency of %s' %title)
    plt.plot(m, eff_p2, '.-', label='p=2')
    plt.plot(m, eff_p4, '.-', label='p=4')
    plt.legend(loc='lower right')
    for M_eff in zip(m, eff_p2):
        plt.annotate('%.2f' % M_eff[1], xy=M_eff)
    for M_eff in zip(m, eff_p4):
        plt.annotate('%.2f' % M_eff[1], xy=M_eff)
    plt.savefig('./pics/eff_%s.pdf' %op)
    plt.clf()

def timing_plot(op, title, ylim, data_s, data_p2, data_p4):
    M = range(7,15)
    plt.xlim([6.5, 14.5])
    # plt.ylim(ylim)
    plt.xlabel('$m$ ($M=2^m$, grid size is $(M+2)\\times(M+2)$ )')
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.plot(M, data_p2, label='2 cores')
    plt.plot(M, data_p4, label='4 cores')
    plt.plot(M, data_s, label='serial')
    plt.legend(loc='upper left')
    plt.savefig('./pics/%s.pdf'%op)
    plt.clf()

eff_plot('sparse', 'sparse solver of diffusion equation', eff_spar_p2, eff_spar_p4)
eff_plot('stepping', 'stepping solver of diffusion equation', eff_step_p2, eff_step_p4)


timing_plot('sparse', 'sparse solver of diffusion eqn', [], par_spar_s, par_spar_p2, par_spar_p4)
timing_plot('stepping', 'stepping solver of diffusion eqn', [], par_step_s, par_step_p2, par_step_p4)