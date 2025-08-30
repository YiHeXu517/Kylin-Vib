/* program-1: auto-correlation functions of N-site naive polariton model */
#include "../include/lattice.hpp"
#include "../include/nil_tdvp.hpp"
int main(int argc, char ** argv)
{
    cout << 0 << endl;
    Lattice lat(argv[1]);
    cout << 1 << endl;
    MPO<MKL_Complex16> ham = lat.gen_ctotal_para();
    cout << 2 << endl;
    vector<size_t> cfg(lat.nsites,0);
    MPS<MKL_Complex16> state0(lat.nsites,lat.nphysdims,cfg);
    cout << 3 << endl;
    TDVP<MKL_Complex16> tdvp(ham,state0);
    MKL_Complex16 dt(0,-0.2);
    tdvp.two_site_evolve(dt,1000,15);
    for(size_t t=0;t<1000;++t)
    {
        cout << tdvp.snaps_[0].overlap(tdvp.snaps_[t]) << endl;
    }
    return 0;
}