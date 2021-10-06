import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# abstract_datall = pd.read_csv(r'C:\Users\74308\OneDrive\wendang\d-delta-smo.csv', encoding='utf-8')
abstract_datal = pd.read_csv(r'C:\Users\74308\OneDrive\wendang\italy-3-fold-delta.csv', encoding='utf-8')
abstract_data = pd.read_csv(r'C:\Users\74308\OneDrive\wendang\italy-3-fold.csv', encoding='utf-8')
states= abstract_datal['State'].tolist()[:21]
renkou=abstract_datal['population'].tolist()[:21]
renkou=np.array(renkou)
renkou=renkou.astype(np.float)
N=21
train_data_length=70
test_data_length=30
date = abstract_datal['Date'].tolist()[::21]
delta=[]
rrr=[]
III=[]
RR=[0.341866494,0.54973822,0.378210806,0.485795455,0.579774692,0.623239437,0.313332399,0.512703136,0.443693444,0.35499769,0.345821326,0.702765875,0.704566636,0.395250271,0.259916006,0.49850075,0.294898883,0.447408928,0.870646766,0.776909722,0.597236356]
II=[0.002298,0.000651,0.000562,0.000784,0.006014,0.002529,0.001241,0.005405,0.008163,0.00416,0.001086,0.005057,0.008106,0.006398,0.001048,0.000796,0.000656,0.00261,0.00155,0.009001,0.003775]
# delta=[0.10001501047263758, 0.21384348659209843, 0.15121195691125672, 0.14294440880708792, 0.1857709875939211, 0.24821801518035858, 0.08194919870045013, 0.18643410666999874, 0.13430212758681956, 0.12701668266320326, 0.10669961377597778, 0.2534345727109864, 0.26180093562608436, 0.1325417058998899, 0.09199791889855011, 0.16796009144162702, 0.09625047072058505, 0.16841676309857906, 0.3851395368255016, 0.31232932266072666, 0.21525055336729623]



# def getll_states(name):
#     dfll = abstract_datall[abstract_datal['State'] == name]
#     return dfll


def getl_states(name):
    dfl = abstract_datal[abstract_datal['State'] == name]
    return dfl


def get_states(name):
    df = abstract_data[abstract_data['State'] == name]
    return df


def predict_delta(state):
    g_state=get_states(state)
    R_true = g_state['recoverrate'].tolist()
    X=np.array(R_true).reshape(-1,2)
    delta_range=np.linspace(0.3,0.4,100)
    Y=delta_range.T
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    scores = -1 * cross_val_score(model, X, Y, cv=3, scoring='neg_mean_squared_error')
    print("MSE scores:\n", scores)
    Rou_opt_I=np.argmin(scores)
    gh=scores[Rou_opt_I]
    return gh


def predict_gh():
    i = 0
    b = []
    while i <= 20:
        a = predict_delta(states[i])
        b.append(a)
        i+=1
    return b


def predict_delta_o():
    i=0
    j=0
    b=[]
    delta=[]
    while i <= 20:
        a = predict_delta(states[i])
        b.append(a)
        i+=1
    while j<=20:
        c=b[j]
        d=[c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c]
        gl_state = getl_states(states[j])
        Rl_true = gl_state['recoverrate'].tolist()
        MSE_delta = mean_squared_error(Rl_true, d)
        delta.append(MSE_delta)
        j+=1
    return delta

delta=predict_delta_o()
rrr=predict_gh()


def SIRconnection(state):
    R=[]
    g_state=getl_states(state)
    hb=states.index(state)
    deltab=delta[hb]
    I_true=g_state['confirml'].tolist()
    for k in range(21):
        if k==0:
            R[k]=0
        else:
            R[k]=R[k-1]+deltab*I_true[k-1]
    return R

def construct_V(state):
    hb=states.index(state)
    deltab=delta[hb]
    n=train_data_length
    g_state = getl_states(state)
    I_true = g_state['confirml'].tolist()
    V=np.ones((n-1,1))
    for k in range(n-1):
        V[k,0]=I_true[k+1]-(1-deltab*I_true[k])
    return V


def construct_F(state):
    col=0
    n=train_data_length
    N=21
    g_state = getl_states(state)
    S= g_state['srate'].tolist()[:n-1]
    I=np.ones((n-1,N))
    F=np.ones((n-1,N))
    for state in states:
        g_state = getl_states(state)
        I_true=g_state['confirml'].tolist()[:n-1]
        I[:,col]=I_true
        col+=1
    for k in range(n-1):
        for i in range(N):
            F[k,i]=S[k]*I[k,i]
    return F


def fun(beta,  i):
    sum = 0
    for k in range(21):
        if i != k:
            sum += beta[k]
    V = construct_V(states[i])
    F = construct_F(states[i])
    u=states.index(states[i])
    eq = np.linalg.norm(V - np.dot(F, np.array(beta).reshape(21, 1)), ord=2) ** 2 + u * sum
    return eq


def estimate_beta(state):

    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 0},
            {'type': 'ineq', 'fun': lambda x: x[1] - 0},
            {'type': 'ineq', 'fun': lambda x: x[2] - 0},
            {'type': 'ineq', 'fun': lambda x: x[3] - 0},
            {'type': 'ineq', 'fun': lambda x: x[4] - 0},
            {'type': 'ineq', 'fun': lambda x: x[5] - 0},
            {'type': 'ineq', 'fun': lambda x: x[6] - 0},
            {'type': 'ineq', 'fun': lambda x: x[7] - 0},
            {'type': 'ineq', 'fun': lambda x: x[8] - 0},
            {'type': 'ineq', 'fun': lambda x: x[9] - 0},
            {'type': 'ineq', 'fun': lambda x: x[10] - 0},
            {'type': 'ineq', 'fun': lambda x: x[11] - 0},
            {'type': 'ineq', 'fun': lambda x: x[12] - 0},
            {'type': 'ineq', 'fun': lambda x: x[13] - 0},
            {'type': 'ineq', 'fun': lambda x: x[14] - 0},
            {'type': 'ineq', 'fun': lambda x: x[15] - 0},
            {'type': 'ineq', 'fun': lambda x: x[16] - 0},
            {'type': 'ineq', 'fun': lambda x: x[17] - 0},
            {'type': 'ineq', 'fun': lambda x: x[18] - 0},
            {'type': 'ineq', 'fun': lambda x: x[19] - 0},
            {'type': 'ineq', 'fun': lambda x: x[20] - 0},


            {'type': 'ineq', 'fun': lambda x: 1 - x[0]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[1]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[2]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[3]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[4]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[5]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[6]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[7]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[8]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[9]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[10]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[11]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[12]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[13]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[14]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[15]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[16]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[17]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[18]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[19]},
            {'type': 'ineq', 'fun': lambda x: 1 - x[20]},

            )
    # x0 = np.array((0.0022978834,0.0006509296,0.0005615938,0.0007842489,0.0060144049,0.0025288114,0.0012407824,0.0054045428,0.0081628432,0.0041600861,0.0010856948,0.0050565638,0.0081064185,0.0063976325,0.0010477709,0.0007962225,0.0006561948,0.0026095537,0.0015499231,0.0090006329,0.0037750132))
    x0=np.ones(21)

    BT = minimize(fun, x0, args=(state), method='SLSQP', constraints=cons)
    print(BT.x)
    return BT

def predict_result(train_data_length, test_data_length):
    I = np.zeros((train_data_length, N))
    I_col=0
    R_col=0
    IT_col=0
    for state in states:
        g_state=getl_states(state)
        I_true = g_state['confirml'].tolist()[:train_data_length]
        I[:,I_col]=I_true
        I_col+=1

    I_test = np.zeros((train_data_length + test_data_length, N))
    for state in states:
        g_state=getl_states(state)
        I_testa=g_state['confirml'].tolist()[:train_data_length + test_data_length]
        I_test[:,IT_col]=I_testa
        IT_col+=1

    R = np.zeros((train_data_length, N))
    for state in states:
        g_state=getl_states(state)
        R_true=g_state['recoverrate'].tolist()[:train_data_length]
        R[:,R_col]=R_true
        R_col+=1

    BT = []
    for state in range(21):
         BT.append(estimate_beta(state))

    I_DDD = []
    I_state = []
    R_state = []
    for j in range(21):
        III= (1 - delta[j]) * 1.02*I[I.shape[0]-1, j] + (1 - 1.02*I[I.shape[0]-1, j] - 1.03*R[R.shape[0] - 1, j]) *sum([BT[j].x[n] * 1.02*I[I.shape[0]-1, j]for n in range(21)])
        I_state.append(III)
        RRR=R[R.shape[0] - 1, j]*1.03+delta[j]*I[I.shape[0]-1, j]*1.02
        R_state.append(RRR)
    I_DDD .append(I_state)
    return I_DDD , I_test


def run_result( Index, length):
    I_test = np.empty
    I_state = []
    while(Index < 100):
        x= predict_result(Index, length)
        for i in range(len(x[0])):
            I_state.append(x[0][i])
        I_test = x[1]
        Index+=length
    return I_state, I_test


def draw_plot():
    pre_result = run_result(70, 1)
    pre_y = np.array(pre_result[0]).T
    real_y = np.array(pre_result[1]).T
    index=[4,6,8]
    for i in index:
        plt.plot(date[:100], real_y[i], marker='o', markersize=5, c='blue',
             label='true data' )
        plt.plot(date[70:], pre_y[i],  marker='d',
              markersize=5, c='red', label='NIPA predict')
        plt.ylabel('感染人口比例', fontsize='12')
        plt.legend()
        plt.grid(linestyle='-')
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))
        plt.xticks(rotation=10)
        # t='this is result'
        # for a, b in zip(real_y[i]*renkou[i], pre_y[i]*renkou[i]):  # 显示数据标签
        #     plt.text(a, b ,t, ha='center', va='bottom', fontsize=7)
        plt.title(states[i] + "", transform=ax.transAxes, fontdict={'size': '14', 'color': 'black'})
        plt.savefig(states[i] + '.png', dpi=600)
        plt.show()
draw_plot()