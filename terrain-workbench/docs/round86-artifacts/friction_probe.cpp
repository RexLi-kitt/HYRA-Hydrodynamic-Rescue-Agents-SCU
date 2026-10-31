#include <cmath>
#include <iostream>
#include <iomanip>
#define NUMERIC_TYPE double
#define C(x) (x)
#define FABS std::abs
#define SQRT std::sqrt
struct Solver {double DepthThresh=1e-3,SpeedThresh=1e-9,g=9.80655,Tstep;};
namespace dg2 {void friction(Solver*,double,double&,double&,double);}
void dg2::friction
(
	Solver *Solverptr,
	NUMERIC_TYPE H,
	NUMERIC_TYPE& HU,
	NUMERIC_TYPE& HV,
	NUMERIC_TYPE n
)
{
	if (H <= Solverptr->DepthThresh)
	{
		HU = C(0.0);
		HV = C(0.0);
		return;
	}

	NUMERIC_TYPE U = HU/H;
	NUMERIC_TYPE V = HV/H;

	if (FABS(U) <= Solverptr->SpeedThresh && FABS(V) <= Solverptr->SpeedThresh)
	{
		return;
	}

	NUMERIC_TYPE speed = SQRT(U*U+V*V);
	NUMERIC_TYPE Cf = Solverptr->g*n*n / pow(H, C(1.0)/C(3.0));
	NUMERIC_TYPE Sfx = -Cf*U*speed;
	NUMERIC_TYPE Sfy = -Cf*V*speed;
	NUMERIC_TYPE Dx = C(1.0) + Solverptr->Tstep*Cf/H * (C(2.0)*U*U + V*V)/speed;
	NUMERIC_TYPE Dy = C(1.0) + Solverptr->Tstep*Cf/H * (U*U + C(2.0)*V*V)/speed;
	HU += Solverptr->Tstep * Sfx / Dx;
	HV += Solverptr->Tstep * Sfy / Dy;
}

int main(){ double h,x,y,n,dt; std::cout<<std::setprecision(17);
while(std::cin>>h>>x>>y>>n>>dt){ Solver s;s.Tstep=dt;dg2::friction(&s,h,x,y,n);std::cout<<x<<" "<<y<<"\n"; } }
