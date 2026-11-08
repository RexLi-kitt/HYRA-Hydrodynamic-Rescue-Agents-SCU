#include <cmath>
#include <algorithm>
#include <array>
// Inputs are the frozen cohort priors, path geometry, observed drift and original bias.
extern "C" __declspec(dllexport) void mobile(int nr,int steps,int nc,const double* poses,
 const double* heads,const double* shifts,const double* masses,const double* bias,const double* cost,
 const double* flow,const double* axis,double easy,int exact,double* result){
 std::array<int,10000> seen{}; int marker=0;
 for(int r=0;r<nr;r++){
  double value=0;
  for(int c=0;c<nc;c++){
   marker++;
   int gx=(int)std::nearbyint(shifts[c*2]/10),gy=(int)std::nearbyint(shifts[c*2+1]/10);
   double sx=exact?shifts[2*c]:gx*10.,sy=exact?shifts[2*c+1]:gy*10.;
   double discount=1;
   for(int t=0;t<steps;t++,discount*=.97){
    int k=2*(r*steps+t);
    double ox=poses[k]-5*(t+1)*flow[0]-sx,oy=poses[k+1]-5*(t+1)*flow[1]-sy;
    int bx=(int)std::floor(ox/10),by=(int)std::floor(oy/10);
    for(int y=std::max(0,by-2);y<=std::min(99,by+2);y++)
     for(int x=std::max(0,bx-2);x<=std::min(99,bx+2);x++){
      int id=100*y+x,wx=x+gx,wy=y+gy;
      if(seen[id]==marker||wx<0||wx>99||wy<0||wy>99)continue;
      double dx=10*x+5-ox,dy=10*y+5-oy;
      if(dx*dx+dy*dy>400.+1e-7||dx*heads[k]+dy*heads[k+1]<-1e-7)continue;
      seen[id]=marker;
      int ix=std::clamp((int)std::nearbyint(dx*axis[0]+dy*axis[1]+20),0,40);
      int iy=std::clamp((int)std::nearbyint(-dx*axis[1]+dy*axis[0]+20),0,40);
      value+=discount*masses[10000*c+id]*bias[100*wy+wx]*(1+easy*(2.8/(2+cost[41*iy+ix])-1));
     }
   }
  }
  result[r]=value;
 }
}
