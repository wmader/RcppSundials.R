/**
** Code auto-generated by cOde 0.2.2.
** Do not edit by hand.
**/
#include <array>
#include <vector>
#include <cmath>
using std::array;
using std::vector;



extern "C" {

/** Derivatives **/
array<vector<double>, 2> dynamics(const double& time, const vector<double>& y,
                                const vector<double>& p, const vector<double>& f) {
    vector<double> ydot(y.size());
    ydot[0] = 1*(0)-1*(p[0]*y[0])-1*(p[0]*y[0]+p[1]*y[1]);
    ydot[1] = 1*(p[0]*y[0])-1*(p[0]*y[0]+p[1]*y[1]);
    ydot[2] = 1*(p[0]*y[0]+p[1]*y[1])-1*(p[2]*y[2]);
    ydot[3] = 1*(p[2]*y[2]);
    array<vector<double>, 2> output = {{ydot, ydot}};
    return output;
}


/** Derivatives of sensitivities **/
vector<double> sensitivities (const double& time,
                              const vector<double>& y, const vector<double>& yS,
                              const vector<double>& p) {
    vector<double> ySdot(y.size() * (y.size() + p.size()));
    ySdot[0] = (-(p[0]+p[0]))*(yS[0])+(-p[1])*(yS[1]);
    ySdot[1] = (p[0]-p[0])*(yS[0])+(-p[1])*(yS[1]);
    ySdot[2] = (p[0])*(yS[0])+(p[1])*(yS[1])+(-p[2])*(yS[2]);
    ySdot[3] = (p[2])*(yS[2]);
    ySdot[4] = (-(p[0]+p[0]))*(yS[4])+(-p[1])*(yS[5]);
    ySdot[5] = (p[0]-p[0])*(yS[4])+(-p[1])*(yS[5]);
    ySdot[6] = (p[0])*(yS[4])+(p[1])*(yS[5])+(-p[2])*(yS[6]);
    ySdot[7] = (p[2])*(yS[6]);
    ySdot[8] = (-(p[0]+p[0]))*(yS[8])+(-p[1])*(yS[9]);
    ySdot[9] = (p[0]-p[0])*(yS[8])+(-p[1])*(yS[9]);
    ySdot[10] = (p[0])*(yS[8])+(p[1])*(yS[9])+(-p[2])*(yS[10]);
    ySdot[11] = (p[2])*(yS[10]);
    ySdot[12] = (-(p[0]+p[0]))*(yS[12])+(-p[1])*(yS[13]);
    ySdot[13] = (p[0]-p[0])*(yS[12])+(-p[1])*(yS[13]);
    ySdot[14] = (p[0])*(yS[12])+(p[1])*(yS[13])+(-p[2])*(yS[14]);
    ySdot[15] = (p[2])*(yS[14]);
    ySdot[16] = (-(p[0]+p[0]))*(yS[16])+(-p[1])*(yS[17])+-(y[0]+y[0]);
    ySdot[17] = (p[0]-p[0])*(yS[16])+(-p[1])*(yS[17])+y[0]-y[0];
    ySdot[18] = (p[0])*(yS[16])+(p[1])*(yS[17])+(-p[2])*(yS[18])+y[0];
    ySdot[19] = (p[2])*(yS[18]);
    ySdot[20] = (-(p[0]+p[0]))*(yS[20])+(-p[1])*(yS[21])+-y[1];
    ySdot[21] = (p[0]-p[0])*(yS[20])+(-p[1])*(yS[21])+-y[1];
    ySdot[22] = (p[0])*(yS[20])+(p[1])*(yS[21])+(-p[2])*(yS[22])+y[1];
    ySdot[23] = (p[2])*(yS[22]);
    ySdot[24] = (-(p[0]+p[0]))*(yS[24])+(-p[1])*(yS[25]);
    ySdot[25] = (p[0]-p[0])*(yS[24])+(-p[1])*(yS[25]);
    ySdot[26] = (p[0])*(yS[24])+(p[1])*(yS[25])+(-p[2])*(yS[26])+-y[2];
    ySdot[27] = (p[2])*(yS[26])+y[2];
    return ySdot;
}


/** Jacobian **/
vector<double> dynamicsJac(const double& time, const std::vector<double>& y, 
                        const std::vector<double>& p,
                        const std::vector<double>& f) {
    vector<double> yJac(y.size()*y.size());
    yJac[0] = -(p[0]+p[0]);
    yJac[1] = p[0]-p[0];
    yJac[2] = p[0];
    yJac[4] = -p[1];
    yJac[5] = -p[1];
    yJac[6] = p[1];
    yJac[10] = -p[2];
    yJac[11] = p[2];
    return yJac;
}


}
