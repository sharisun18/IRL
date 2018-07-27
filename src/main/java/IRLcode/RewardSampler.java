package IRLcode;

import burlap.behavior.singleagent.learnfromdemo.mlirl.differentiableplanners.DifferentiableVI;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;


public interface RewardSampler {

    RealVector[] sampleRewards(int Nrf, RealMatrix[] TransP, RealMatrix Ppi);

    boolean isValid(RealVector newR, RealMatrix[] TransP, RealMatrix Ppi);
}
