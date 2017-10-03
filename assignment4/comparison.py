import matplotlib.pyplot as plt

import laplace
import rr


def compare_plots(n=100, beta=0.05, epsilon=0.5):
    error_rr, calculated_beta_rr, alpha_rr = rr.accuracy(epsilon, n, beta)
    error_lp, calculated_beta_lp, alpha_lp = laplace.accuracy(epsilon, n, beta)

    print('Laplace calculated beta = {}'.format(calculated_beta_lp))
    print('RR calculated beta = {}'.format(calculated_beta_rr))
    f, axarr = plt.subplots(2, sharex=True)
    # laplace
    axarr[0].set_title('Laplace')
    axarr[0].axhline(0, color='r')
    alpha_line_lp = axarr[0].axhline(alpha_lp, color='r')
    alpha_line2_lp = axarr[0].axhline(-alpha_lp, color='r')
    axarr[0].set_xlabel('Nth run')
    axarr[0].set_ylabel('Error')
    axarr[0].legend([alpha_line_lp],
                    ['alpha = {:.6f}'.format(alpha_lp)])
    axarr[0].plot(error_lp, 'go')
    # rr
    axarr[1].set_title('Randomized Response')
    axarr[1].axhline(0, color='g')
    alpha_line_rr = axarr[1].axhline(alpha_rr, color='r')
    alpha_line2_rr = axarr[1].axhline(-alpha_rr, color='r')
    axarr[1].plot(error_rr, 'go')
    axarr[1].set_xlabel('Nth run')
    axarr[1].set_ylabel('Error')
    axarr[1].legend([alpha_line_rr],['alpha = '+'{:.6f}'.format(alpha_rr)])

    plt.show()

if __name__ == '__main__':
    compare_plots()
