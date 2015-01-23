import matplotlib.pyplot as plt
import numpy as np

M = range(6,15)

mvm_p2 = [6.55584335327e-05, 7.48419761658e-05, 0.000110285997391, 0.000251077651978,
0.000838329315186, 0.00312665081024, 0.0110255544186, 0.0426148719788, 0.170487000704]
mvm_p4 = [0.000278891444206, 0.000278010368347, 0.000266078472137, 0.000429366111755,
0.00091201543808, 0.00235075688362, 0.00847489452362, 0.0323157141209, 0.122092933655]
mvm_s = [5.02872467041e-06, 1.86228752136e-05, 6.97429180145e-05, 0.000271816015244,
0.00111368513107, 0.00430407333374, 0.0172973926067, 0.0698734884262, 0.276121017456]

mma_p2 = [8.64386558533e-05, 0.000221304893494, 0.000494011163712, 0.0015133600235,
0.00462313199043, 0.0169877066612, 0.0662298984528, 0.26448877573]
mma_p4 = [0.000267244577408, 0.000490869760513, 0.000687114953995, 0.00173725986481,
0.00491859555244, 0.0197650139332, 0.0794573254585, 0.310671880245]
mma_s = [3.79490852356e-06, 1.67863368988e-05, 5.98740577698e-05, 0.000507966995239,
0.0019399330616, 0.011117395401, 0.0423836541176, 0.177587729692]

mmm_p2 = [0.000198786263953, 0.0013297533989, 0.0100760941505, 0.0816765546799,
0.622372784615, 5.09899855614]
mmm_p4 = [0.000143371582031, 0.00091305398941, 0.00664101743698, 0.0583698837757,
0.465134063482, 3.84114269257]
mmm_s = [0.000264954566956, 0.00187545776367, 0.0132710576057, 0.103937907219,
0.859698805809, 7.27340384722]

fft_p2 = [0.000203566551208, 0.000502159595489, 0.00127208471298, 0.00556966781616,
0.0199775862694, 0.086671807766, 0.322147285938]
fft_p4 = [0.000384466648102, 0.000765972137452, 0.00146064281464, 0.00478273630142,
0.0168445968628, 0.0695847296715, 0.28141317606]
fft_s  = [5.35559654236e-05, 0.000227725505829, 0.000906374454498, 0.00472079753876,
0.0197847056389, 0.0853590703011, 0.364001245499]

def eff(t1, tP, p):
	# return np array of values that correspond to the efficiencies of parallel operations
	# efficiency is defined as: E_p = t1 / (p * tP)
	# t1 = list of serial times
	# tP = list of parallel times
	# p  = number of processors
	return np.array(t1) / (p*np.array(tP))

eff_mvm_p2 = eff(mvm_s, mvm_p2, 2)
eff_mvm_p4 = eff(mvm_s, mvm_p4, 4)
eff_mma_p2 = eff(mma_s, mma_p2, 2)
eff_mma_p4 = eff(mma_s, mma_p4, 4)
eff_mmm_p2 = eff(mmm_s, mmm_p2, 2)
eff_mmm_p4 = eff(mmm_s, mmm_p4, 4)
eff_fft_p2 = eff(fft_s, fft_p2, 2)
eff_fft_p4 = eff(fft_s, fft_p4, 4)

def eff_plot(op, title, eff_p2, eff_p4):
	m = M[:len(eff_p2)]
	plt.xlim([5.5, 14.5])
	plt.xlabel('$M$ ($N=2^M$, size of matrix)')
	plt.ylabel('Efficiency')
	plt.title('Efficiency of %s' %title)
	plt.plot(m, eff_p2, '.-', label='p=2')
	plt.plot(m, eff_p4, '.-', label='p=4')
	plt.legend(loc='lower right')
	for M_eff in zip(m, eff_p2):
		plt.annotate('%.2f' % M_eff[1], xy=M_eff)
	for M_eff in zip(m, eff_p4):
		plt.annotate('%.2f' % M_eff[1], xy=M_eff)
	plt.savefig('eff_%s.pdf' %op)
	plt.clf()

eff_plot('mvm', 'matrix-vector multiplication', eff_mvm_p2, eff_mvm_p4)
eff_plot('mma', 'matrix-matrix addition', eff_mma_p2, eff_mma_p4)
eff_plot('mmm', 'matrix-matrix multiplication', eff_mmm_p2, eff_mmm_p4)
eff_plot('fft', 'FFT', eff_fft_p2, eff_fft_p4)


# plt.xlim([5.5,14.5])
# plt.ylim([-0.01, 0.3])
# plt.xlabel('$M$ ($N=2^M$, size of matrix)')
# plt.ylabel('Time (s)')
# plt.title('Matrix-Vector Multiplication')
# plt.plot(M, mvm_p2, label='2 cores')
# plt.plot(M, mvm_p4, label='4 cores')
# plt.plot(M, mvm_s, label='serial')
# plt.legend(loc='upper left')
# # plt.show()
# plt.savefig('mvm.pdf')
# plt.clf()

# plt.xlim([5.5,13.5])
# plt.ylim([-0.01, 0.35])
# plt.xlabel('$M$ ($N=2^M$, size of matrix)')
# plt.ylabel('Time (s)')
# plt.title('Matrix-Matrix Addition')
# plt.plot(M[:len(mma_p2)], mma_p2, label='2 cores')
# plt.plot(M[:len(mma_p2)], mma_p4, label='4 cores')
# plt.plot(M[:len(mma_p2)], mma_s, label='serial')
# plt.legend(loc='upper left')
# # plt.show()
# plt.savefig('mma.pdf')
# plt.clf()

# plt.xlim([5.5,11.5])
# plt.ylim([-0.10, 8])
# plt.xlabel('$M$ ($N=2^M$, size of matrix)')
# plt.ylabel('Time (s)')
# plt.title('Matrix-Matrix Multiplication')
# plt.plot(M[:len(mmm_p2)], mmm_p2, label='2 cores')
# plt.plot(M[:len(mmm_p2)], mmm_p4, label='4 cores')
# plt.plot(M[:len(mmm_p2)], mmm_s, label='serial')
# plt.legend(loc='upper left')
# # plt.show()
# plt.savefig('mmm.pdf')
# plt.clf()

# plt.xlim([5.5,12.5])
# plt.ylim([-0.01, 0.4])
# plt.xlabel('$M$ ($N=2^M$, size of matrix)')
# plt.ylabel('Time (s)')
# plt.title('FFT of a Matrix')
# plt.plot(M[:len(fft_p2)], fft_p2, label='2 cores')
# plt.plot(M[:len(fft_p2)], fft_p4, label='4 cores')
# plt.plot(M[:len(fft_p2)], fft_s, label='serial')
# plt.legend(loc='upper left')
# # plt.show()
# plt.savefig('fft.pdf')
# plt.clf()