#include "../hls/conv.hpp"
#include <fstream>
#include <iostream>
int main(int argc,char**argv) {
 if(argc!=3){std::cerr<<"usage: tb_conv vectors.txt output.txt\n";return 2;}
 std::ifstream in(argv[1]); if(!in){return 2;}
 int8_hls x[H*H],w[C]; int32_hls y[P]; int v;
 for(auto &e:x){if(!(in>>v)||v<-128||v>127)return 2;e=v;}
 for(auto &e:w){if(!(in>>v)||v<-128||v>127)return 2;e=v;}
 conv_int8(x,w,y);
 std::ofstream out(argv[2]);if(!out)return 2;
 for(auto e:y)out<<static_cast<int>(e)<<'\n';
}
