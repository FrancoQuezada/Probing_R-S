"""Parameterized copy of the 002/003 routing model; originals stay untouched.
Only model parameters are injected. The objective remains max_i E_z[Y_i]-alpha'z.
"""
module ProbingRegimeModel
include(joinpath(@__DIR__,"probing_experiments_common.jl"))
using .ProbingExperiments
using Random, Statistics, LinearAlgebra, SHA, TOML
const E=ProbingExperiments
const C=E.C

Base.@kwdef struct ModelConfig
    p_complex::NTuple{4,Float64}=(.02,.12,.45,.70)
    p_hit0::NTuple{4,Float64}=(.60,.65,.75,.92)
    p_false0::NTuple{4,Float64}=(.05,.08,.15,.20)
    beta::Float64=2.20
    mean_interarrival::Float64=.80
    fast_mean::NTuple{2,Float64}=(.75,2.40)
    std_mean::NTuple{2,Float64}=(1.,8.)
    revenue_scale::Float64=8.
    fast_routing_cost::Float64=.70
    delay_penalty::Float64=.42
    alpha::NTuple{2,Float64}=(10.,4.5)
end

function validate(cfg::ModelConfig)
    all(x->isfinite(x)&&0<=x<=1,[cfg.p_complex...;cfg.p_hit0...;cfg.p_false0...]) || error("Invalid probabilities")
    isfinite(cfg.beta)&&cfg.beta>=0 || error("Invalid signal beta")
    all(x->isfinite(x)&&x>0,[cfg.mean_interarrival;cfg.fast_mean...;cfg.std_mean...]) || error("Invalid time means")
    all(x->isfinite(x)&&x>=0,[cfg.revenue_scale;cfg.fast_routing_cost;cfg.delay_penalty;cfg.alpha...]) || error("Invalid economic parameters")
    return cfg
end

function model_dict(cfg::ModelConfig)
    return Dict("P_COMPLEX"=>collect(cfg.p_complex),"P_HIT0"=>collect(cfg.p_hit0),
        "P_FALSE0"=>collect(cfg.p_false0),"BETA"=>cfg.beta,"MEAN_INTERARRIVAL"=>cfg.mean_interarrival,
        "FAST_MEAN"=>collect(cfg.fast_mean),"STD_MEAN"=>collect(cfg.std_mean),
        "REVENUE_SCALE"=>cfg.revenue_scale,"FAST_ROUTING_COST"=>cfg.fast_routing_cost,
        "DELAY_PENALTY"=>cfg.delay_penalty,"ALPHA"=>collect(cfg.alpha),
        "NCLASS"=>4,"ACTIVE_CLASSES"=>[3,4])
end
function digest(x::AbstractDict)
    io=IOBuffer();TOML.print(io,x;sorted=true);return bytes2hex(sha256(take!(io)))
end
model_digest(cfg)=digest(model_dict(cfg))
function scenarios()
    [(id="S$a$b$c",A=a,B=b,C=c,model=validate(ModelConfig(
        p_hit0=a==0 ? (.60,.65,.75,.92) : (.60,.65,.55,.65),
        p_false0=a==0 ? (.05,.08,.15,.20) : (.05,.08,.20,.25),
        mean_interarrival=b==0 ? .80 : .60,fast_routing_cost=c==0 ? .70 : 1.40)))
        for a in 0:1 for b in 0:1 for c in 0:1]
end

p_hit(c,z,cfg::ModelConfig)=cfg.p_hit0[c]+(1.0-cfg.p_hit0[c])*(1.0-exp(-cfg.beta*Float64(z)))
p_false(c,z,cfg::ModelConfig)=cfg.p_false0[c]*exp(-cfg.beta*Float64(z))
signal_probability(c,complex,z,cfg)=complex ? p_hit(c,z,cfg) : p_false(c,z,cfg)
cost(cfg,z)=dot(collect(cfg.alpha),z)

"""Exactly the original primitive RNG draw order, with explicit parameters."""
function primitive_path(rng,cfg::ModelConfig;njobs=45)
    jobs=Vector{C.CRNJobPrimitive}(undef,njobs);t=0.
    for r in 1:njobs
        t+=C.randexp_mean(rng,cfg.mean_interarrival)
        c=rand(rng,1:4);d=rand(rng)<cfg.p_complex[c]
        basev=.15+.16*c;v=clamp(basev+.35*(rand(rng)-.5),.05,1.)
        muf=d ? cfg.fast_mean[2] : cfg.fast_mean[1]
        mus=d ? cfg.std_mean[2] : cfg.std_mean[1]
        sf=C.randexp_mean(rng,muf);ss=C.randexp_mean(rng,mus);u=rand(rng)
        jobs[r]=C.CRNJobPrimitive(t,c,d,v,sf,ss,u)
    end
    return C.CRNPath(jobs)
end
function primitive_bank(cfg,n,seed;njobs=45)
    rng=MersenneTwister(seed);[primitive_path(rng,cfg;njobs) for _ in 1:n]
end
function materialize(path,z,cfg)
    zz=C.full_z(z);jobs=Vector{C.JobPrimitive}(undef,length(path.jobs))
    for (r,j) in enumerate(path.jobs)
        s=j.u_signal<signal_probability(j.class,j.complex,zz[j.class],cfg) ? 1 : 0
        jobs[r]=C.JobPrimitive(j.arrival,j.class,j.complex,j.value,j.service_fast,j.service_std,s)
    end
    return C.SamplePath(jobs)
end
materialize_bank(bank,z,cfg)=[materialize(p,z,cfg) for p in bank]

function evaluate_policy(policy,path,cfg::ModelConfig)
    free_fast=0.;free_std=0.;total=0.
    for job in path.jobs
        t=job.arrival;qf=max(free_fast-t,0.);qs=max(free_std-t,0.)
        score=policy.av*job.value-policy.aq*(qf-qs)+policy.as*job.signal
        route_fast=score>=policy.tau
        if route_fast
            start=max(t,free_fast);finish=start+job.service_fast;free_fast=finish
            time_in_system=finish-t
            total+=cfg.revenue_scale*job.value
            total-=cfg.fast_routing_cost
            total-=cfg.delay_penalty*time_in_system
        else
            start=max(t,free_std);finish=start+job.service_std;free_std=finish
            time_in_system=finish-t
            total+=cfg.revenue_scale*job.value
            total-=cfg.delay_penalty*time_in_system
        end
    end
    return total
end
function rewards(policies,bank,cfg)
    Y=Matrix{Float64}(undef,length(policies),length(bank))
    for i in eachindex(policies),l in eachindex(bank)
        Y[i,l]=evaluate_policy(policies[i],bank[l],cfg)
    end
    return Y
end

"""Cache repeated scalar log probabilities/derivatives, keeping the ORIGINAL
job-by-job summation order. No sufficient-count summation or LR normalization.
"""
function distribution_terms(z,cfg)
    zz=C.full_z(z);logs=zeros(8);scores=zeros(8)
    for pos in 1:2, complex in (true,false)
        c=pos+2
        p=clamp(signal_probability(c,complex,zz[c],cfg),1e-12,1.0-1e-12)
        dp=complex ? (1.0-cfg.p_hit0[c])*cfg.beta*exp(-cfg.beta*zz[c]) :
                     -cfg.p_false0[c]*cfg.beta*exp(-cfg.beta*zz[c])
        idx=(pos-1)*4+(complex ? 0 : 2)
        logs[idx+1]=log(p);logs[idx+2]=log1p(-p)
        scores[idx+1]=dp/p;scores[idx+2]=-dp/(1.0-p)
    end
    return logs,scores
end
function lr_kernel(z,zsource,cfg)
    lt,s=distribution_terms(z,cfg);ls,_=distribution_terms(zsource,cfg)
    return lt-ls,s
end
function lr_with_kernel(path,kernel)
    logs,sc=kernel;ll=0.;score=zeros(2)
    for job in path.jobs
        pos=job.class==3 ? 1 : job.class==4 ? 2 : 0
        pos==0 && continue
        idx=(pos-1)*4+(job.complex ? 0 : 2)+(job.signal==1 ? 1 : 2)
        ll+=logs[idx];score[pos]+=sc[idx]
    end
    return ll,score
end
loglr_and_score(path,z,zsource,cfg)=lr_with_kernel(path,lr_kernel(z,zsource,cfg))
function bank_scores(bank,z,zsource,cfg)
    kernel=lr_kernel(z,zsource,cfg);n=length(bank);ll=zeros(n);S=zeros(2,n)
    for l in 1:n
        ll[l],s=lr_with_kernel(bank[l],kernel);S[:,l].=s
    end
    return ll,S
end
function local_data(z,bank,Y,cfg)
    n=length(bank);ll,S=bank_scores(bank,z,z,cfg)
    all(==(0.),ll) || error("Fresh LR differs from one")
    values=vec(mean(Y;dims=2));best=argmax(values);G=(Y*transpose(S))/n
    hard=[mean(view(Y,best,:).*view(S,j,:))-cfg.alpha[j] for j in 1:2]
    return (values=values,G=G,hard=hard,best=best,scores=S,ess=Float64(n),banks=1,lr_clips=0)
end

struct HistoricalBank
    scenario::String
    model_sha256::String
    zgen::Vector{Float64}
    bank::Vector{C.SamplePath}
    Y::Matrix{Float64}
end
function historical_bank(scenario,cfg,z,bank,Y)
    HistoricalBank(scenario,model_digest(cfg),copy(z),bank,Y)
end
function green_data(z,history,cfg,scenario)
    T=length(history);m=size(history[1].Y,1);values=zeros(m);G=zeros(m,2)
    Ls=Vector{Vector{Float64}}();Ss=Matrix{Float64}[];sumL=0.;sumL2=0.;clips=0
    expected=model_digest(cfg)
    for hb in history
        hb.scenario==scenario && hb.model_sha256==expected || error("Cross-scenario/configuration Green history")
        n=length(hb.bank);ll,S=bank_scores(hb.bank,z,hb.zgen,cfg)
        clips+=count(x->abs(x)>35,ll);L=exp.(clamp.(ll,-35.,35.))
        push!(Ls,L);push!(Ss,S)
        for i in 1:m;values[i]+=mean(view(hb.Y,i,:).*L)/T;end
        G .+=(hb.Y*transpose(S.*transpose(L)))/(n*T)
        sumL+=sum(L);sumL2+=sum(abs2,L)
    end
    best=argmax(values);hard=-collect(cfg.alpha)
    for (t,hb) in enumerate(history),j in 1:2
        hard[j]+=mean(view(hb.Y,best,:).*Ls[t].*view(Ss[t],j,:))/T
    end
    return (values=values,G=G,hard=hard,best=best,ess=sumL^2/sumL2,banks=T,lr_clips=clips)
end
combine(G,w,cfg)=transpose(G)*w-collect(cfg.alpha)
softmax_weights(values;beta=.5)=C.softmax_weights(values;beta)
bootstrap_weights(Y,B,rng)=E.bootstrap_weights(Y,B,rng)
projected_step(z,g,k)=C.projected_step(z,g,k)

"""Routing under each policy's own evolving queues, on the SAME jobs.
Delay penalty is the original time-in-system charge, not a due-date tardiness.
"""
function routing_details(policy,path,cfg)
    ff=0.;fs=0.;wait=0.;fast_busy=0.;delay_cost=0.;route_cost=0.;profit=0.;decisions=BitVector()
    for job in path.jobs
        t=job.arrival;qf=max(ff-t,0.);qs=max(fs-t,0.)
        fast=policy.av*job.value-policy.aq*(qf-qs)+policy.as*job.signal>=policy.tau
        push!(decisions,fast)
        if fast
            start=max(t,ff);finish=start+job.service_fast;ff=finish
            fast_busy+=job.service_fast;route_cost+=cfg.fast_routing_cost
            profit+=cfg.revenue_scale*job.value;profit-=cfg.fast_routing_cost
        else
            start=max(t,fs);finish=start+job.service_std;fs=finish
            profit+=cfg.revenue_scale*job.value
        end
        wait+=start-t;charge=cfg.delay_penalty*(finish-t);delay_cost+=charge;profit-=charge
    end
    horizon=max(ff,fs,path.jobs[end].arrival)
    return (decisions=decisions,fast_fraction=count(decisions)/length(decisions),
        fast_utilization=fast_busy/horizon,total_wait=wait,delay_penalty=delay_cost,
        routing_cost=route_cost,reward=profit,horizon=horizon)
end
end
