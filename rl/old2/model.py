import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

class FourierTimeEmbedding(nn.Module):
    """ 시간 정보를 푸리에 변환을 사용하여 임베딩하는 모듈
    시간 정보를 주기적인 특성을 가진 고차원 벡터로 변환합니다.
    
    Args:
        embed_dim (int): 출력 임베딩 차원
        num_bands (int): 푸리에 변환에 사용할 주파수 밴드 수
    """
    def __init__(self,embed_dim=64,num_bands=16):
        super().__init__()
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        self.fc = nn.Linear(2*num_bands,embed_dim)
    def forward(self,t):
        # t: [batch,seq,1]
        coeffs = torch.linspace(0,1,self.num_bands,device=t.device)
        embed = torch.einsum('bs,n->bsn',t,coeffs) * 2 * torch.pi
        # embed = t*coeffs[None,None,:]*2*torch.pi
        embed = torch.cat([torch.sin(embed),torch.cos(embed)],dim=-1)
        return self.fc(embed)

class ExecutionHyridModule(nn.Module):
    """ 1분봉 데이터를 처리하는 CNN-GRU 하이브리드 모듈
    CNN으로 지역적 특징을 추출하고 GRU로 시계열 정보를 처리합니다.
    
    Args:
        input_dim (int): 입력 특징 차원
        time_dim (int): 시간 임베딩 차원
        hidden_size (int): 은닉층 크기
    """
    def __init__(self,input_dim,time_dim=64,hidden_size=32):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(time_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim+time_dim,hidden_size,5,padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size)
        )
        self.gru = nn.GRU(hidden_size,hidden_size,batch_first=True)
        self.proj = nn.Linear(hidden_size, 64)
    def forward(self,x,t):
        # x: [B,T,F], t: [B,T,1]
        t_emb = self.time_embed(t)
        x = torch.cat([x,t_emb],dim=-1).permute(0,2,1)
        conv_out = self.conv(x).permute(0,2,1)
        gru_out,_ = self.gru(conv_out)
        return self.proj(gru_out)

class MultiScaleLSTM(nn.Module):
    """ 15분/4시간 봉 데이터를 위한 멀티스케일 LSTM 모듈
    여러 시간 스케일에서 LSTM을 적용하여 다양한 시간대의 패턴을 포착합니다.
    
    Args:
        input_dim (int): 입력 특징 차원
        time_dim (int): 시간 임베딩 차원
        scales (list): 각 LSTM이 처리할 시간 스케일 목록
    """
    def __init__(self,input_dim,time_dim=64,scales=[5,10,20]):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(time_dim)
        self.lstms = nn.ModuleList([
            nn.LSTM(input_dim+time_dim,32,batch_first=True)
            for _ in scales
        ])
        self.attn = nn.MultiheadAttention(32*len(scales),4,batch_first=True)
        self.proj = nn.Linear(32*len(scales),64)
    def forward(self,x,t):
        t_emb = self.time_embed(t)
        x_in = torch.cat([x,t_emb],dim=-1)
        outputs = []
        for lstm in self.lstms:
            out,_ = lstm(x_in)
            outputs.append(out)
        concat = torch.cat(outputs,dim=-1)
        attn_out,_ = self.attn(concat,concat,concat)
        return self.proj(attn_out)

class HierarchicalTransformer(nn.Module):
    """ 1시간/일 봉 데이터를 처리하는 트랜스포머 기반 모듈
    자기 주의 메커니즘을 통해 장기 의존성을 포착합니다.
    
    Args:
        input_dim (int): 입력 특징 차원
        time_dim (int): 시간 임베딩 차원
        nhead (int): 멀티헤드 어텐션의 헤드 수
        num_layers (int): 트랜스포머 인코더 층 수
    """
    def __init__(self,input_dim,time_dim=64,nhead=4,num_layers=2):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(time_dim)
        self.input_proj = nn.Linear(input_dim+time_dim,64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,nhead=nhead,dim_feedforward=256,
            activation='gelu',batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers)
    def forward(self,x,t):
        t_emb = self.time_embed(t)
        x_in = torch.cat([x,t_emb],dim=-1)
        projected = self.input_proj(x_in)
        return self.encoder(projected)

class CrossModalAttention(nn.Module):
    """ 다중 시간대 특징을 통합하는 교차 모달 어텐션 모듈
    서로 다른 시간대의 특징들 간의 관계를 학습합니다.
    
    Args:
        num_timeframes (int): 처리할 시간대 수
        embed_dim (int): 특징 임베딩 차원
        heads (int): 어텐션 헤드 수
    """
    def __init__(self, input_dims, embed_dim=64, heads=4):
        super().__init__()
        self.timeframes = input_dims.keys()
        num_timeframes = len(self.timeframes)
        self.projections = nn.ModuleList([
            nn.Linear(dim,embed_dim)
            for tf, dim in input_dims.items()
        ])
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, heads, batch_first=True)
            for _ in range(num_timeframes)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_timeframes)
        ])
        self.timeframe_weights = nn.Parameter(torch.ones(num_timeframes)/num_timeframes)
        self.final_norm = nn.LayerNorm(embed_dim * num_timeframes)
    def forward(self, features):
        features = [proj(feat.transpose(1,2)).transpose(1,2)
            for proj,feat in zip(self.projections,features)]
        # [B, K, T, D] 형태로 변환
        context = torch.stack(features, dim=1)
        B, K, T, D = context.shape
        # 배치와 시퀀스 차원 결합
        context = context.view(B*T, K, D)
        # 각 시간대별 어텐션 적용
        attn_outs = []
        for i, (attn, norm) in enumerate(zip(self.attentions, self.norms)):
            # 현재 시간대를 쿼리로 사용
            query = context[:, i:i+1]
            # 어텐션 및 잔차 연결
            out, _ = attn(query, context, context)
            attn_outs.append(norm(out + query))
        # 모든 시간대의 특징 결합
        weights = F.softmax(self.timeframe_weights,dim=0)
        combined = torch.stack(attn_outs,dim=1)*weights.view(1,K,1,1)
        fused = combined.view(B,T,K*D)
        return self.final_norm(fused)

class EnhancedMultiTimeframeModel(nn.Module):
    """ 다중 시간대 데이터를 처리하는 강화학습 모델
    각 시간대별로 특화된 모듈을 사용하여 특징을 추출하고,
    이를 통합하여 행동(actor)과 가치(critic) 예측을 수행합니다.
    
    Args:
        feature_dims (dict): 각 시간대별 입력 특징 차원을 담은 딕셔너리
    """
    def __init__(self, feature_dims, input_dims, action_dim):
        super().__init__()
        # 입력된 시간대 저장
        self.timeframes = list(feature_dims.keys())
        
        # 각 시간대별 특화된 모듈 초기화
        self.modules_dict = nn.ModuleDict()
        for tf, dim in feature_dims.items():
            # 시간대에 따라 적절한 모듈 선택
            if int(tf) <= 20:  # 1분봉
                self.modules_dict[tf] = ExecutionHyridModule(dim)
            elif int(tf) <= 100:  # 1시간 이하
                self.modules_dict[tf] = MultiScaleLSTM(dim)
            else:  # 1시간 초과
                self.modules_dict[tf] = HierarchicalTransformer(dim)
        
        # 특징 융합을 위한 어텐션 모듈
        self.fusion = CrossModalAttention(
            input_dims,
            embed_dim=64  # 각 모듈의 출력 차원
        )
        
        # 행동과 가치 예측을 위한 헤드
        fusion_dim = 64 * len(self.timeframes)  # 융합된 특징의 차원
    
        # 행동 (롱/숏/중립)
        self.actor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, action_dim)  # 3개의 행동: [롱, 중립, 숏]
        )
        
        # Critic 네트워크 (가치 예측)
        self.critic = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
    
    def forward(self, inputs):
        features = []
        # 입력된 모든 시간대에 대해 처리
        for tf in self.timeframes:
            if tf not in inputs:
                raise ValueError(f"입력에 시간대 {tf}가 없습니다.")
            data, time = inputs[tf]
            out = self.modules_dict[tf](data, time)
            features.append(out)
        # 특징 융합
        fused = self.fusion(features)
        logits = self.actor(fused)
        dist = Categorical(logits=logits)
        value = self.critic(fused)
        return dist, value
    def get_action(self,inputs,deterministic=False,mode='last'):
        dist,value = self.forward(inputs)
        if deterministic:
            action = torch.argmax(dist.probs,dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        if mode == 'last':
            return action[:,-1],log_prob[:,-1],entropy[:,-1],value[:,-1]
        elif mode == 'mean':
            return action[:,-1],log_prob.mean(dim=1),entropy.mean(dim=1),value.mean(dim=1,keepdim=True)
        else:
            return action,log_prob,entropy,value
    
    def get_logprob(self,inputs,action,mode='last'):
        dist,value = self.forward(inputs)
        B,T,_ = dist.probs.shape
        action = action.unsqueeze(1).expand(B, T)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
    if mode == 'last':
        return log_prob[:, -1], entropy[:, -1], value[:, -1]
    elif mode == 'mean':
        return log_prob.mean(dim=1), entropy.mean(dim=1), value.mean(dim=1, keepdim=True)
    else:
        return log_prob,entropy,value

model = EnhancedMultiTimeframeModel(feature_dims,input_dims,3)
model(x)