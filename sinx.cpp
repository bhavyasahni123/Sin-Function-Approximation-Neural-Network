#include<iostream>
#include<iomanip>
#include<cmath>
#include<random>
#include<vector>
using namespace std;
#define _USE_MATH_DEFINES



double sigmoid(double x){
    return 1/(1+exp(-x));
}

int main(){

    vector<double> inputs;
    vector<double> outputs;

    double lower_limit = -M_PI;
    double upper_limit = M_PI;

    uniform_real_distribution<double> unif(lower_limit, upper_limit);
    default_random_engine re(1);

    for(int j=0;j<50000;j++){
        double x = unif(re);
        double y = sin(x);

        inputs.push_back(x);
        outputs.push_back(y);
    }

    double w1=0.1,w2=0.2,w3=-0.1,w4=-0.2,w5=0.3;
    double w01=0.1,w02=-0.1,w03=0.2,w04=-0.2,w05=0.3;

    double v1=0.1,v2=0.2,v3=-0.1,v4=-0.2,v5=0.3;
    double v0=0.1;

    double a1,a2,a3,a4,a5,a6;
    double z1,z2,z3,z4,z5;
    double error;
    double y_hat=0;

    double n=0.1;

    int N=inputs.size();

    for(int epoch=0;epoch<100000;epoch++){

        double dw1=0,dw2=0,dw3=0,dw4=0,dw5=0;
        double db1=0,db2=0,db3=0,db4=0,db5=0;

        double dv1=0,dv2=0,dv3=0,dv4=0,dv5=0;
        double db0=0;

        for(int i=0;i<N;i++){

            a1=w1*inputs[i]+w01;
            a2=w2*inputs[i]+w02;
            a3=w3*inputs[i]+w03;
            a4=w4*inputs[i]+w04;
            a5=w5*inputs[i]+w05;

            z1=sigmoid(a1);
            z2=sigmoid(a2);
            z3=sigmoid(a3);
            z4=sigmoid(a4);
            z5=sigmoid(a5);

            a6=v1*z1+v2*z2+v3*z3+v4*z4+v5*z5+v0;

            y_hat=a6;

            error=outputs[i]-y_hat;

            double delta1=error*v1*z1*(1-z1);
            double delta2=error*v2*z2*(1-z2);
            double delta3=error*v3*z3*(1-z3);
            double delta4=error*v4*z4*(1-z4);
            double delta5=error*v5*z5*(1-z5);

            dw1+=delta1*inputs[i];
            dw2+=delta2*inputs[i];
            dw3+=delta3*inputs[i];
            dw4+=delta4*inputs[i];
            dw5+=delta5*inputs[i];

            db1+=delta1;
            db2+=delta2;
            db3+=delta3;
            db4+=delta4;
            db5+=delta5;

            dv1+=error*z1;
            dv2+=error*z2;
            dv3+=error*z3;
            dv4+=error*z4;
            dv5+=error*z5;

            db0+=error;
        }

        
        double SSE=0;

        for(int i=0;i<N;i++){

            a1=w1*inputs[i]+w01;
            a2=w2*inputs[i]+w02;
            a3=w3*inputs[i]+w03;
            a4=w4*inputs[i]+w04;
            a5=w5*inputs[i]+w05;

            z1=sigmoid(a1);
            z2=sigmoid(a2);
            z3=sigmoid(a3);
            z4=sigmoid(a4);
            z5=sigmoid(a5);

            a6=v1*z1+v2*z2+v3*z3+v4*z4+v5*z5+v0;

            y_hat=a6;

            error=outputs[i]-y_hat;

            SSE+=error*error;
        }
        w1+=n*dw1/N;
        w2+=n*dw2/N;
        w3+=n*dw3/N;
        w4+=n*dw4/N;
        w5+=n*dw5/N;

        w01+=n*db1/N;
        w02+=n*db2/N;
        w03+=n*db3/N;
        w04+=n*db4/N;
        w05+=n*db5/N;

        v1+=n*dv1/N;
        v2+=n*dv2/N;
        v3+=n*dv3/N;
        v4+=n*dv4/N;
        v5+=n*dv5/N;

        v0+=n*db0/N;


        double MSE=SSE/N;

        cout<<fixed<<setprecision(6);
        cout<<"Epoch: "<<epoch <<" | MSE: "<<MSE<<endl;

        if(MSE<0.00001){
             break;
        }
    }

    cout<<"\nFinal weights after training:\n";

    cout<<"w1: "<<w1
        <<" w2: "<<w2
        <<" w3: "<<w3
        <<" w4: "<<w4
        <<" w5: "<<w5<<endl;

    cout<<"w01: "<<w01
        <<" w02: "<<w02
        <<" w03: "<<w03
        <<" w04: "<<w04
        <<" w05: "<<w05<<endl;

    cout<<"v1: "<<v1
        <<" v2: "<<v2
        <<" v3: "<<v3
        <<" v4: "<<v4
        <<" v5: "<<v5
        <<" v0: "<<v0<<endl;

    return 0;
}
