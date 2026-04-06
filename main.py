"""
DAO Ethics Engine

Curated public edition of the Multi-Agent AI Consensus + Ethics node used around
DAO Hub in the broader Sampo AI OS ecosystem.
"""

import random
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import bcrypt
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError
from jose import jwt as jose_jwt
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Settings(BaseSettings):
    SECRET_KEY: str = "dev-only-change-me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    DATABASE_URL: str = "sqlite:///./consensus.db"
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "change-me"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

engine_kwargs: Dict[str, Any] = {}
if settings.DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="admin")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Agent(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    agent_type = Column(String, default="learner")
    state = Column(String, default="idle")
    belief = Column(Float, default=0.5)
    reward_total = Column(Float, default=0.0)
    message_count = Column(Integer, default=0)
    capabilities = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ConsensusRound(Base):
    __tablename__ = "consensus_rounds"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    round_number = Column(Integer, nullable=False)
    algorithm = Column(String, nullable=False)
    participating_agents = Column(JSON, nullable=False)
    initial_beliefs = Column(JSON, nullable=False)
    final_consensus = Column(Float, nullable=True)
    converged = Column(String, default="false")
    iterations = Column(Integer, default=0)
    consensus_error = Column(Float, nullable=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    num_agents = Column(Integer, nullable=False)
    algorithm = Column(String, nullable=False)
    byzantine_fraction = Column(Float, default=0.0)
    results = Column(JSON, nullable=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)


class ProposalEvaluation(Base):
    __tablename__ = "proposal_evaluations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    proposal_name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    beneficiary_distribution = Column(JSON, nullable=False)
    beneficiary_labels = Column(JSON, nullable=True)
    evidence_claims = Column(JSON, nullable=False)
    public_benefit_score = Column(Float, nullable=False)
    gini_coefficient = Column(Float, nullable=False)
    evidence_quality_score = Column(Float, nullable=False)
    bias_flags = Column(JSON, nullable=False)
    overall_ethics_score = Column(Float, nullable=False)
    recommendation = Column(String, nullable=False)
    reasoning = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AgentCreate(BaseModel):
    name: str
    agent_type: str = "learner"
    capabilities: List[str] = Field(default_factory=lambda: ["consensus", "learning"])


class AgentResponse(BaseModel):
    id: str
    name: str
    agent_type: str
    state: str
    belief: float
    reward_total: float
    message_count: int
    capabilities: List[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class ConsensusRequest(BaseModel):
    algorithm: str = "belief_propagation"
    max_iterations: int = 100
    convergence_threshold: float = 0.01
    byzantine_fraction: float = 0.0


class ConsensusRoundResponse(BaseModel):
    id: str
    round_number: int
    algorithm: str
    participating_agents: List[str]
    initial_beliefs: Dict[str, float]
    final_consensus: Optional[float] = None
    converged: str
    iterations: int
    consensus_error: Optional[float] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class SimulationRequest(BaseModel):
    name: str
    num_agents: int = 10
    algorithm: str = "belief_propagation"
    byzantine_fraction: float = 0.0
    max_iterations: int = 200


class SimulationResponse(BaseModel):
    id: str
    name: str
    num_agents: int
    algorithm: str
    byzantine_fraction: float
    results: Optional[Dict[str, Any]] = None
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class NetworkTopologyResponse(BaseModel):
    total_agents: int
    active_agents: int
    byzantine_agents: int
    leader_id: Optional[str] = None
    consensus_rounds: int
    network_connectivity: float
    fault_tolerance: float


class EvidenceClaim(BaseModel):
    claim: str
    sample_size: int = 0
    confidence: float = 0.0
    source: str = "unspecified"
    peer_reviewed: bool = False


class ProposalEvaluationRequest(BaseModel):
    proposal_name: str
    description: Optional[str] = None
    beneficiary_distribution: List[float]
    beneficiary_labels: Optional[List[str]] = None
    evidence_claims: List[EvidenceClaim] = Field(default_factory=list)


class ProposalEvaluationResponse(BaseModel):
    id: str
    proposal_name: str
    description: Optional[str] = None
    beneficiary_distribution: List[float]
    beneficiary_labels: Optional[List[str]] = None
    evidence_claims: List[dict]
    public_benefit_score: float
    gini_coefficient: float
    evidence_quality_score: float
    bias_flags: List[dict]
    overall_ethics_score: float
    recommendation: str
    reasoning: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class EthicsAuditSummary(BaseModel):
    total_proposals: int
    approved: int
    needs_review: int
    rejected: int
    avg_public_benefit_score: float
    avg_gini_coefficient: float
    avg_evidence_quality: float
    avg_overall_ethics_score: float
    top_bias_types: List[str]
    ethics_trend: str


AGENT_VALUES = {
    "primary_mandate": "Maximise benefit to the broadest possible population",
    "decision_basis": "Statistical evidence and verifiable empirical data only",
    "prohibited": [
        "Political agenda alignment or ideological framing",
        "Serving a minority elite without measurable broader public benefit",
        "Amplifying cognitive biases instead of correcting them",
        "Accepting claims without statistical foundation (n<30 or CI<0.80)",
        "Elite capture: benefit Gini > 0.40 triggers automatic rejection",
    ],
    "required_thresholds": {
        "min_public_benefit_ratio": 0.50,
        "max_gini_concentration": 0.40,
        "min_evidence_quality": 0.60,
        "min_statistical_confidence": 0.80,
        "min_sample_size": 30,
    },
    "bias_watchlist": [
        "confirmation_bias",
        "availability_heuristic",
        "anchoring_bias",
        "elite_capture",
        "groupthink",
        "survivorship_bias",
        "status_quo_bias",
        "appeal_to_authority",
        "false_dichotomy",
        "overfitting_to_recent_data",
    ],
    "fairness_model": "Rawlsian veil of ignorance",
}


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jose_jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    return jose_jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])


def compute_gini(values: List[float]) -> float:
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    arr = np.array(sorted(values), dtype=float)
    arr = arr / total
    n = len(arr)
    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) * arr).sum() / n)


def evaluate_evidence_quality(claims: List[dict]) -> dict:
    if not claims:
        return {
            "score": 0.0,
            "breakdown": [],
            "warning": "No evidence provided - proposal cannot be approved",
        }

    scored_claims = []
    for claim in claims:
        sample_size = claim.get("sample_size", 0)
        confidence = claim.get("confidence", 0.0)
        peer_reviewed = claim.get("peer_reviewed", False)
        source = claim.get("source", "unspecified")

        if sample_size >= 30:
            sample_score = min(1.0, max(0.0, np.log10(max(sample_size, 1)) / 3.0))
        else:
            sample_score = max(0.0, sample_size / 60.0)

        confidence_score = max(0.0, (confidence - 0.5) / 0.5) if confidence >= 0.5 else 0.0
        source_bonus = 0.15 if peer_reviewed else (0.05 if source != "unspecified" else 0.0)
        claim_score = min(1.0, sample_score * 0.5 + confidence_score * 0.4 + source_bonus)

        scored_claims.append(
            {
                "claim": claim.get("claim", ""),
                "sample_size": int(sample_size),
                "confidence": float(confidence),
                "peer_reviewed": bool(peer_reviewed),
                "score": round(float(claim_score), 3),
                "meets_threshold": claim_score >= AGENT_VALUES["required_thresholds"]["min_evidence_quality"],
            }
        )

    aggregate_score = float(np.mean([item["score"] for item in scored_claims]))
    weak_claims = [item for item in scored_claims if not item["meets_threshold"]]
    return {
        "score": round(aggregate_score, 3),
        "breakdown": scored_claims,
        "weak_claims_count": len(weak_claims),
        "warning": f"{len(weak_claims)} claim(s) below evidence threshold" if weak_claims else None,
    }


def detect_cognitive_biases(distribution: List[float], claims: List[dict], description: str = "") -> List[dict]:
    flags: List[dict] = []

    if distribution and max(distribution) > 0.60:
        flags.append(
            {
                "bias_type": "elite_capture",
                "severity": "high",
                "description": (
                    f"Top beneficiary segment receives {max(distribution) * 100:.1f}% of total benefit. "
                    "Rawlsian threshold: <=40%."
                ),
            }
        )

    sources = [claim.get("source", "unspecified") for claim in claims]
    if len(sources) > 2 and len(set(sources)) == 1:
        flags.append(
            {
                "bias_type": "groupthink",
                "severity": "medium",
                "description": "All evidence claims cite the same source. Independent replication is required.",
            }
        )

    for claim in claims:
        if claim.get("confidence", 0.0) > 0.95 and claim.get("sample_size", 0) < 50:
            flags.append(
                {
                    "bias_type": "survivorship_bias",
                    "severity": "medium",
                    "description": (
                        f"Claim '{claim.get('claim', '')}': confidence {claim.get('confidence')} with "
                        f"n={claim.get('sample_size')} is statistically suspect."
                    ),
                }
            )
            break

    if len(distribution) == 1:
        flags.append(
            {
                "bias_type": "confirmation_bias",
                "severity": "high",
                "description": "A single-segment distribution ignores population diversity.",
            }
        )

    if not claims:
        flags.append(
            {
                "bias_type": "status_quo_bias",
                "severity": "high",
                "description": "No evidence claims submitted. Unsupported proposals cannot be approved.",
            }
        )

    distribution_sum = sum(distribution)
    if distribution and abs(distribution_sum - 1.0) > 0.05:
        flags.append(
            {
                "bias_type": "anchoring_bias",
                "severity": "low",
                "description": f"Beneficiary distribution sums to {distribution_sum:.3f}, not 1.0. Auto-normalising.",
            }
        )

    description_lower = description.lower()
    if "inevitable" in description_lower or "obvious" in description_lower:
        flags.append(
            {
                "bias_type": "appeal_to_authority",
                "severity": "low",
                "description": "Proposal description relies on asserted certainty instead of measured evidence.",
            }
        )

    return flags


def compute_public_benefit_score(distribution: List[float]) -> float:
    if not distribution:
        return 0.0
    total = sum(distribution)
    if total == 0:
        return 0.0

    normalised = [value / total for value in distribution]
    equal_share = 1.0 / len(normalised)
    covered_segments = sum(1 for value in normalised if value >= equal_share * 0.5)
    coverage_ratio = covered_segments / len(normalised)
    gini = compute_gini(normalised)
    equality_bonus = max(0.0, 0.2 * (1.0 - gini / 0.4))
    return min(1.0, round(coverage_ratio + equality_bonus, 3))


def evaluate_proposal(
    proposal_name: str,
    description: str,
    beneficiary_distribution: List[float],
    evidence_claims: List[dict],
) -> dict:
    del proposal_name
    thresholds = AGENT_VALUES["required_thresholds"]
    distribution_total = sum(beneficiary_distribution) if beneficiary_distribution else 0
    if distribution_total > 0 and abs(distribution_total - 1.0) > 0.0001:
        beneficiary_distribution = [value / distribution_total for value in beneficiary_distribution]

    gini = compute_gini(beneficiary_distribution)
    public_benefit = compute_public_benefit_score(beneficiary_distribution)
    evidence = evaluate_evidence_quality(evidence_claims)
    bias_flags = detect_cognitive_biases(beneficiary_distribution, evidence_claims, description)

    gini_score = max(0.0, 1.0 - (gini / thresholds["max_gini_concentration"]))
    overall = round(public_benefit * 0.40 + evidence["score"] * 0.35 + gini_score * 0.25, 3)

    rejection_triggers = []
    if gini > thresholds["max_gini_concentration"]:
        rejection_triggers.append(
            f"Gini={gini:.3f} exceeds max allowed {thresholds['max_gini_concentration']} - potential elite capture"
        )
    if public_benefit < thresholds["min_public_benefit_ratio"]:
        rejection_triggers.append(
            f"Public benefit coverage {public_benefit:.2f} < required {thresholds['min_public_benefit_ratio']}"
        )
    if evidence["score"] < thresholds["min_evidence_quality"] or not evidence_claims:
        rejection_triggers.append(
            f"Evidence quality {evidence['score']:.2f} < required {thresholds['min_evidence_quality']}"
        )

    high_severity_biases = [flag for flag in bias_flags if flag["severity"] == "high"]
    if high_severity_biases:
        rejection_triggers.append(
            f"{len(high_severity_biases)} high-severity bias(es) detected: "
            f"{', '.join(flag['bias_type'] for flag in high_severity_biases)}"
        )

    if rejection_triggers:
        recommendation = "reject"
    elif overall >= 0.75 and not bias_flags:
        recommendation = "approve"
    elif overall >= 0.60:
        recommendation = "review"
    else:
        recommendation = "reject"

    return {
        "public_benefit_score": public_benefit,
        "gini_coefficient": round(gini, 4),
        "evidence_quality_score": evidence["score"],
        "bias_flags": bias_flags,
        "overall_ethics_score": overall,
        "recommendation": recommendation,
        "reasoning": {
            "gini_score_component": round(gini_score, 3),
            "coverage_score_component": public_benefit,
            "evidence_score_component": evidence["score"],
            "evidence_detail": evidence,
            "rejection_triggers": rejection_triggers,
            "agent_values_applied": list(thresholds.keys()),
            "fairness_model": AGENT_VALUES["fairness_model"],
        },
    }


def run_belief_propagation(beliefs: Dict[str, float], max_iter: int, threshold: float) -> dict:
    agents = list(beliefs.keys())
    values = {agent_id: belief for agent_id, belief in beliefs.items()}

    if len(agents) < 2:
        consensus_value = list(values.values())[0] if values else 0.0
        return {"consensus": consensus_value, "converged": True, "iterations": 1, "error": 0.0}

    for iteration in range(1, max_iter + 1):
        next_values = {}
        for agent_id in agents:
            neighbour_pool = [candidate for candidate in agents if candidate != agent_id]
            neighbour_count = min(len(neighbour_pool), max(2, len(agents) // 3))
            neighbours = random.sample(neighbour_pool, neighbour_count)
            neighbour_values = [values[neighbour] for neighbour in neighbours]
            next_values[agent_id] = 0.7 * values[agent_id] + 0.3 * (sum(neighbour_values) / len(neighbour_values))

        max_change = max(abs(next_values[agent_id] - values[agent_id]) for agent_id in agents)
        values = next_values
        if max_change < threshold:
            consensus_value = sum(values.values()) / len(values)
            return {
                "consensus": consensus_value,
                "converged": True,
                "iterations": iteration,
                "error": float(np.std(list(values.values()))),
            }

    consensus_value = sum(values.values()) / len(values)
    return {
        "consensus": consensus_value,
        "converged": False,
        "iterations": max_iter,
        "error": float(np.std(list(values.values()))),
    }


def run_raft_consensus(beliefs: Dict[str, float], max_iter: int) -> dict:
    del max_iter
    agents = list(beliefs.keys())
    leader = random.choice(agents)
    leader_value = beliefs[leader]
    final_values = {agent_id: leader_value * (1 + random.gauss(0, 0.002)) for agent_id in agents}
    return {
        "consensus": sum(final_values.values()) / len(final_values),
        "converged": True,
        "iterations": random.randint(3, 8),
        "error": float(np.std(list(final_values.values()))),
        "leader": leader,
    }


def run_byzantine_ft(beliefs: Dict[str, float], byzantine_fraction: float, max_iter: int, threshold: float) -> dict:
    agents = list(beliefs.keys())
    byzantine_count = int(len(agents) * byzantine_fraction)
    byzantine_agents = set(random.sample(agents, byzantine_count)) if byzantine_count else set()
    honest_agents = [agent_id for agent_id in agents if agent_id not in byzantine_agents]

    values = {agent_id: belief for agent_id, belief in beliefs.items()}
    for agent_id in byzantine_agents:
        values[agent_id] = random.uniform(0.0, 1.0)

    for iteration in range(1, max_iter + 1):
        updated_values = {}
        for agent_id in honest_agents:
            ordered_values = sorted(values.values())
            trim = max(1, byzantine_count) if byzantine_count else 0
            trimmed_values = ordered_values[trim : len(ordered_values) - trim] if trim and len(ordered_values) > 2 * trim else ordered_values
            updated_values[agent_id] = sum(trimmed_values) / len(trimmed_values)
        for agent_id in byzantine_agents:
            updated_values[agent_id] = random.uniform(0.0, 1.0)

        max_change = max(abs(updated_values[agent_id] - values[agent_id]) for agent_id in honest_agents) if honest_agents else 0.0
        values = updated_values
        if max_change < threshold:
            honest_values = [values[agent_id] for agent_id in honest_agents] or [0.5]
            return {
                "consensus": sum(honest_values) / len(honest_values),
                "converged": True,
                "iterations": iteration,
                "error": float(np.std(honest_values)),
                "byzantine_detected": byzantine_count,
            }

    honest_values = [values[agent_id] for agent_id in honest_agents] or [0.5]
    return {
        "consensus": sum(honest_values) / len(honest_values),
        "converged": False,
        "iterations": max_iter,
        "error": float(np.std(honest_values)),
        "byzantine_detected": byzantine_count,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    database = SessionLocal()
    try:
        if not database.query(User).filter_by(username=settings.ADMIN_USERNAME).first():
            database.add(
                User(
                    id=str(uuid.uuid4()),
                    username=settings.ADMIN_USERNAME,
                    hashed_password=hash_password(settings.ADMIN_PASSWORD),
                    role="admin",
                )
            )

        if database.query(Agent).count() == 0:
            agent_types = ["coordinator", "validator", "learner", "learner", "learner"]
            for index, agent_type in enumerate(agent_types, start=1):
                database.add(
                    Agent(
                        id=str(uuid.uuid4()),
                        name=f"Agent-{index:03d}",
                        agent_type=agent_type,
                        state="active",
                        belief=round(random.uniform(0.3, 0.8), 3),
                        capabilities=["consensus", "learning", "propagation"],
                    )
                )

        database.commit()
    finally:
        database.close()

    yield


app = FastAPI(
    title="DAO Ethics Engine",
    description="Multi-agent consensus simulation and ethics evaluation node for DAO governance workflows.",
    version="1.0.0",
    lifespan=lifespan,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


def get_db():
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = decode_token(token)
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    user = db.query(User).filter_by(username=username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@app.get("/")
def root() -> dict:
    return {
        "service": "dao-ethics-engine",
        "role": "Multi-Agent AI Consensus + Ethics node",
        "docs": "/docs",
    }


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    return {
        "status": "ok",
        "service": "dao-ethics-engine",
        "agents": db.query(Agent).count(),
        "consensus_rounds": db.query(ConsensusRound).count(),
        "simulations": db.query(SimulationRun).count(),
        "evaluations": db.query(ProposalEvaluation).count(),
    }


@app.post("/api/v1/auth/token", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> dict:
    user = db.query(User).filter_by(username=form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    return {"access_token": create_access_token({"sub": user.username}), "token_type": "bearer"}


@app.get("/api/v1/agents", response_model=List[AgentResponse])
def list_agents(db: Session = Depends(get_db), _user: User = Depends(get_current_user)) -> List[Agent]:
    return db.query(Agent).all()


@app.post("/api/v1/agents", response_model=AgentResponse, status_code=201)
def create_agent(data: AgentCreate, db: Session = Depends(get_db), _user: User = Depends(get_current_user)) -> Agent:
    agent = Agent(
        id=str(uuid.uuid4()),
        name=data.name,
        agent_type=data.agent_type,
        state="active",
        belief=round(random.uniform(0.3, 0.8), 3),
        capabilities=data.capabilities,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


@app.get("/api/v1/agents/{agent_id}", response_model=AgentResponse)
def get_agent(agent_id: str, db: Session = Depends(get_db), _user: User = Depends(get_current_user)) -> Agent:
    agent = db.query(Agent).filter_by(id=agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@app.post("/api/v1/consensus/run", response_model=ConsensusRoundResponse, status_code=201)
def run_consensus(req: ConsensusRequest, db: Session = Depends(get_db), _user: User = Depends(get_current_user)) -> ConsensusRound:
    agents = db.query(Agent).filter_by(state="active").all()
    if len(agents) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 active agents")

    initial_beliefs = {agent.id: agent.belief for agent in agents}
    round_number = db.query(ConsensusRound).count() + 1

    if req.algorithm == "raft":
        result = run_raft_consensus(initial_beliefs, req.max_iterations)
    elif req.algorithm == "byzantine_ft":
        result = run_byzantine_ft(initial_beliefs, req.byzantine_fraction, req.max_iterations, req.convergence_threshold)
    else:
        result = run_belief_propagation(initial_beliefs, req.max_iterations, req.convergence_threshold)

    consensus_value = result["consensus"]
    for agent in agents:
        agent.belief = round(float(consensus_value) + random.gauss(0, 0.01), 4)
        agent.message_count += result["iterations"]
        agent.reward_total += float(1.0 - result["error"]) if result["converged"] else 0.0

    round_record = ConsensusRound(
        id=str(uuid.uuid4()),
        round_number=round_number,
        algorithm=req.algorithm,
        participating_agents=[agent.id for agent in agents],
        initial_beliefs=initial_beliefs,
        final_consensus=round(float(consensus_value), 6),
        converged=str(result["converged"]).lower(),
        iterations=result["iterations"],
        consensus_error=round(float(result["error"]), 6),
        completed_at=datetime.now(timezone.utc),
    )
    db.add(round_record)
    db.commit()
    db.refresh(round_record)
    return round_record


@app.get("/api/v1/consensus/history", response_model=List[ConsensusRoundResponse])
def consensus_history(limit: int = 20, db: Session = Depends(get_db), _user: User = Depends(get_current_user)) -> List[ConsensusRound]:
    return db.query(ConsensusRound).order_by(ConsensusRound.started_at.desc()).limit(limit).all()


@app.post("/api/v1/simulation/run", response_model=SimulationResponse, status_code=201)
def run_simulation(req: SimulationRequest, db: Session = Depends(get_db), _user: User = Depends(get_current_user)) -> SimulationRun:
    if req.num_agents > 100:
        raise HTTPException(status_code=400, detail="Max 100 agents per simulation")

    beliefs = {f"sim_agent_{index}": round(random.uniform(0.1, 0.9), 3) for index in range(req.num_agents)}
    if req.algorithm == "raft":
        result = run_raft_consensus(beliefs, req.max_iterations)
    elif req.algorithm == "byzantine_ft":
        result = run_byzantine_ft(beliefs, req.byzantine_fraction, req.max_iterations, 0.01)
    else:
        result = run_belief_propagation(beliefs, req.max_iterations, 0.01)

    simulation = SimulationRun(
        id=str(uuid.uuid4()),
        name=req.name,
        num_agents=req.num_agents,
        algorithm=req.algorithm,
        byzantine_fraction=req.byzantine_fraction,
        status="completed",
        results={
            "consensus_value": round(float(result["consensus"]), 6),
            "converged": result["converged"],
            "iterations_to_convergence": result["iterations"],
            "final_error": round(float(result["error"]), 6),
            "initial_beliefs_sample": dict(list(beliefs.items())[:5]),
            "algorithm_details": {
                key: value
                for key, value in result.items()
                if key not in ("consensus", "converged", "iterations", "error")
            },
        },
        completed_at=datetime.now(timezone.utc),
    )
    db.add(simulation)
    db.commit()
    db.refresh(simulation)
    return simulation


@app.get("/api/v1/simulation/runs", response_model=List[SimulationResponse])
def list_simulations(db: Session = Depends(get_db), _user: User = Depends(get_current_user)) -> List[SimulationRun]:
    return db.query(SimulationRun).order_by(SimulationRun.created_at.desc()).all()


@app.get("/api/v1/network/topology", response_model=NetworkTopologyResponse)
def network_topology(db: Session = Depends(get_db), _user: User = Depends(get_current_user)) -> NetworkTopologyResponse:
    agents = db.query(Agent).all()
    total_agents = len(agents)
    active_agents = sum(1 for agent in agents if agent.state == "active")
    byzantine_agents = sum(1 for agent in agents if agent.state == "byzantine")
    coordinator = next((agent for agent in agents if agent.agent_type == "coordinator"), None)
    consensus_rounds = db.query(ConsensusRound).count()
    connectivity = min(1.0, active_agents / max(total_agents, 1) * 0.9 + 0.1) if total_agents else 0.0
    fault_tolerance = round((total_agents - byzantine_agents) / max(total_agents, 1), 3) if total_agents else 1.0
    return NetworkTopologyResponse(
        total_agents=total_agents,
        active_agents=active_agents,
        byzantine_agents=byzantine_agents,
        leader_id=coordinator.id if coordinator else None,
        consensus_rounds=consensus_rounds,
        network_connectivity=round(connectivity, 3),
        fault_tolerance=fault_tolerance,
    )


@app.get("/api/v1/learning/rewards")
def learning_rewards(db: Session = Depends(get_db), _user: User = Depends(get_current_user)) -> List[dict]:
    agents = db.query(Agent).order_by(Agent.reward_total.desc()).all()
    return [
        {
            "agent_id": agent.id,
            "name": agent.name,
            "reward_total": round(agent.reward_total, 4),
            "message_count": agent.message_count,
            "current_belief": agent.belief,
        }
        for agent in agents
    ]


@app.get("/api/v1/ethics/values")
def get_agent_values() -> dict:
    return {
        "manifest": AGENT_VALUES,
        "version": "1.0.0",
        "immutable": True,
        "description": "These values govern all agent decisions and proposal reviews.",
    }


@app.post("/api/v1/ethics/evaluate", response_model=ProposalEvaluationResponse, status_code=201)
def evaluate_proposal_endpoint(
    req: ProposalEvaluationRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> ProposalEvaluation:
    if not req.beneficiary_distribution:
        raise HTTPException(status_code=400, detail="beneficiary_distribution cannot be empty")
    if any(value < 0 for value in req.beneficiary_distribution):
        raise HTTPException(status_code=400, detail="beneficiary_distribution values must be non-negative")
    if req.beneficiary_labels and len(req.beneficiary_labels) != len(req.beneficiary_distribution):
        raise HTTPException(status_code=400, detail="beneficiary_labels length must match beneficiary_distribution")

    claims_raw = [claim.model_dump() for claim in req.evidence_claims]
    result = evaluate_proposal(
        proposal_name=req.proposal_name,
        description=req.description or "",
        beneficiary_distribution=req.beneficiary_distribution,
        evidence_claims=claims_raw,
    )

    labels = req.beneficiary_labels or [f"G{index + 1}" for index in range(len(req.beneficiary_distribution))]
    result["reasoning"]["beneficiary_breakdown"] = [
        {"label": labels[index], "share": round(req.beneficiary_distribution[index], 4)}
        for index in range(len(req.beneficiary_distribution))
    ]

    record = ProposalEvaluation(
        id=str(uuid.uuid4()),
        proposal_name=req.proposal_name,
        description=req.description,
        beneficiary_distribution=req.beneficiary_distribution,
        beneficiary_labels=labels,
        evidence_claims=claims_raw,
        public_benefit_score=result["public_benefit_score"],
        gini_coefficient=result["gini_coefficient"],
        evidence_quality_score=result["evidence_quality_score"],
        bias_flags=result["bias_flags"],
        overall_ethics_score=result["overall_ethics_score"],
        recommendation=result["recommendation"],
        reasoning=result["reasoning"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.get("/api/v1/ethics/evaluations", response_model=List[ProposalEvaluationResponse])
def list_evaluations(limit: int = 50, db: Session = Depends(get_db), _user: User = Depends(get_current_user)) -> List[ProposalEvaluation]:
    return db.query(ProposalEvaluation).order_by(ProposalEvaluation.created_at.desc()).limit(limit).all()


@app.get("/api/v1/ethics/audit", response_model=EthicsAuditSummary)
def ethics_audit(db: Session = Depends(get_db), _user: User = Depends(get_current_user)) -> EthicsAuditSummary:
    evaluations = db.query(ProposalEvaluation).all()
    total = len(evaluations)
    if total == 0:
        return EthicsAuditSummary(
            total_proposals=0,
            approved=0,
            needs_review=0,
            rejected=0,
            avg_public_benefit_score=0.0,
            avg_gini_coefficient=0.0,
            avg_evidence_quality=0.0,
            avg_overall_ethics_score=0.0,
            top_bias_types=[],
            ethics_trend="no data",
        )

    approved = sum(1 for evaluation in evaluations if evaluation.recommendation == "approve")
    needs_review = sum(1 for evaluation in evaluations if evaluation.recommendation == "review")
    rejected = sum(1 for evaluation in evaluations if evaluation.recommendation == "reject")

    average_public_benefit = float(np.mean([evaluation.public_benefit_score for evaluation in evaluations]))
    average_gini = float(np.mean([evaluation.gini_coefficient for evaluation in evaluations]))
    average_evidence = float(np.mean([evaluation.evidence_quality_score for evaluation in evaluations]))
    average_ethics = float(np.mean([evaluation.overall_ethics_score for evaluation in evaluations]))

    bias_counter: Dict[str, int] = {}
    for evaluation in evaluations:
        for bias_flag in evaluation.bias_flags or []:
            bias_type = bias_flag.get("bias_type", "unknown")
            bias_counter[bias_type] = bias_counter.get(bias_type, 0) + 1

    top_bias_types = sorted(bias_counter.keys(), key=lambda key: -bias_counter[key])[:5]

    if total >= 6:
        third = max(1, total // 3)
        early_average = float(np.mean([evaluation.overall_ethics_score for evaluation in evaluations[:third]]))
        recent_average = float(np.mean([evaluation.overall_ethics_score for evaluation in evaluations[-third:]]))
        if recent_average > early_average + 0.02:
            trend = "improving"
        elif recent_average < early_average - 0.02:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "insufficient data"

    return EthicsAuditSummary(
        total_proposals=total,
        approved=approved,
        needs_review=needs_review,
        rejected=rejected,
        avg_public_benefit_score=round(average_public_benefit, 3),
        avg_gini_coefficient=round(average_gini, 3),
        avg_evidence_quality=round(average_evidence, 3),
        avg_overall_ethics_score=round(average_ethics, 3),
        top_bias_types=top_bias_types,
        ethics_trend=trend,
    )


@app.post("/api/v1/ethics/gini/calculate")
def calculate_gini(distribution: List[float], _user: User = Depends(get_current_user)) -> dict:
    if not distribution or any(value < 0 for value in distribution):
        raise HTTPException(status_code=400, detail="All values must be >= 0")

    gini_value = compute_gini(distribution)
    threshold = AGENT_VALUES["required_thresholds"]["max_gini_concentration"]
    passes = gini_value <= threshold
    return {
        "gini_coefficient": round(gini_value, 4),
        "threshold": threshold,
        "passes": passes,
        "verdict": (
            "Within acceptable range - benefit is sufficiently distributed"
            if passes
            else f"Exceeds threshold - value {gini_value:.3f} > {threshold}. Proposal would be rejected."
        ),
    }
