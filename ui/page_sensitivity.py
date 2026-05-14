"""Page: Sensitivity Analysis — change one voter's preference and compare results."""

from __future__ import annotations

import streamlit as st

from ..core.models import Voter, Ballot
from ..core.condorcet import compute_full_condorcet
from ..core.borda import compute_borda_scores
from ..core.copeland import compute_copeland_scores
from ..core.consensus import compute_consensus_degree
from ..core.power_index import compute_power_indices


def render_page():
    """Render the sensitivity analysis page."""
    st.header("敏感性分析 — 测试偏好变化的影响")

    ballot = st.session_state.get("ballot")
    if not ballot:
        st.warning("请先在「建模」页面输入或选择一个决策场景。")
        return

    # Select voter to modify
    voter_names = [v.name for v in ballot.voters]
    selected_voter = st.selectbox("选择要修改偏好的投票人", voter_names, key="sens_voter")

    original_voter = next(v for v in ballot.voters if v.name == selected_voter)

    st.subheader(f"当前偏好: {selected_voter}")
    st.write(" > ".join(original_voter.preference))

    # New preference input
    st.subheader("修改偏好")
    new_prefs = []
    cols = st.columns(len(ballot.alternatives))
    for i, alt in enumerate(ballot.alternatives):
        with cols[i]:
            rank = st.number_input(
                f"{alt} 排名",
                min_value=1,
                max_value=len(ballot.alternatives),
                value=original_voter.preference.index(alt) + 1,
                key=f"sens_rank_{alt}",
            )
            new_prefs.append((rank, alt))

    new_prefs.sort()
    new_pref_list = [alt for _, alt in new_prefs]

    # Run comparison
    if st.button("运行敏感性分析", type="primary", key="run_sensitivity"):
        # Original results
        orig_condorcet = compute_full_condorcet(ballot)
        orig_borda = compute_borda_scores(ballot)
        orig_copeland = compute_copeland_scores(ballot)

        # Modified ballot
        new_voter = Voter(name=selected_voter, preference=new_pref_list)
        modified_voters = [new_voter if v.name == selected_voter else v for v in ballot.voters]
        modified_ballot = Ballot(voters=modified_voters, alternatives=ballot.alternatives)

        mod_condorcet = compute_full_condorcet(modified_ballot)
        mod_borda = compute_borda_scores(modified_ballot)
        mod_copeland = compute_copeland_scores(modified_ballot)

        # Display comparison
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("修改前")
            if orig_condorcet["condorcet_winner"]:
                st.success(f"孔多塞赢家: {orig_condorcet['condorcet_winner']}")
            else:
                st.warning("无孔多塞赢家")
            if orig_condorcet["cycle_detected"]:
                st.error(f"循环: {' → '.join(orig_condorcet['cycle_path'])}")
            st.write("**Borda 排名:**")
            for i, alt in enumerate(orig_borda.ranking, 1):
                st.write(f"  {i}. {alt} ({orig_borda.scores[alt]})")

        with col2:
            st.subheader("修改后")
            if mod_condorcet["condorcet_winner"]:
                st.success(f"孔多塞赢家: {mod_condorcet['condorcet_winner']}")
            else:
                st.warning("无孔多塞赢家")
            if mod_condorcet["cycle_detected"]:
                st.error(f"循环: {' → '.join(mod_condorcet['cycle_path'])}")
            st.write("**Borda 排名:**")
            for i, alt in enumerate(mod_borda.ranking, 1):
                st.write(f"  {i}. {alt} ({mod_borda.scores[alt]})")

        # Summary of changes
        st.divider()
        st.subheader("变化分析")

        cycle_broken = orig_condorcet["cycle_detected"] and not mod_condorcet["cycle_detected"]
        cycle_created = not orig_condorcet["cycle_detected"] and mod_condorcet["cycle_detected"]
        winner_changed = orig_condorcet["condorcet_winner"] != mod_condorcet["condorcet_winner"]
        borda_changed = orig_borda.ranking != mod_borda.ranking

        if cycle_broken:
            st.success("循环被打破！偏好变化消除了循环。")
        if cycle_created:
            st.error("循环产生！偏好变化引入了新的循环。")
        if winner_changed:
            st.warning(f"孔多塞赢家变化: {orig_condorcet['condorcet_winner']} → {mod_condorcet['condorcet_winner']}")
        if borda_changed:
            st.info("Borda 排名发生变化。")
        if not any([cycle_broken, cycle_created, winner_changed, borda_changed]):
            st.success("偏好变化未改变最终结果，系统具有一定稳健性。")

        # Consensus sensitivity
        st.divider()
        st.subheader("共识度敏感性分析")

        orig_cd = compute_consensus_degree(ballot)
        mod_cd = compute_consensus_degree(modified_ballot)
        cd_change = mod_cd - orig_cd

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("原始共识度", f"{orig_cd:.2%}")
        with col2:
            st.metric("修改后共识度", f"{mod_cd:.2%}", delta=f"{cd_change:+.2%}")
        with col3:
            if abs(cd_change) < 0.01:
                st.metric("稳健性", "高")
            elif abs(cd_change) < 0.1:
                st.metric("稳健性", "中")
            else:
                st.metric("稳健性", "低")

        if cd_change > 0:
            st.success(f"偏好修改使共识度提高了 {cd_change:.2%}。")
        elif cd_change < 0:
            st.warning(f"偏好修改使共识度降低了 {abs(cd_change):.2%}。")
        else:
            st.info("偏好修改未改变共识度。")

    # Power index sensitivity section
    st.divider()
    st.subheader("投票权力敏感性分析")

    if ballot.num_voters > 10:
        st.info("投票人超过10人时，权力指数计算量过大。")
    else:
        st.write("**模拟移除投票人后对权力分布的影响：**")

        if ballot.num_voters >= 3:
            remove_voter = st.selectbox(
                "选择要模拟移除的投票人",
                [v.name for v in ballot.voters],
                key="power_remove_voter",
            )

            if st.button("计算权力变化", key="run_power_sensitivity"):
                with st.spinner("计算中..."):
                    voter_weights = [v.voting_weight for v in ballot.voters]
                    orig_power = compute_power_indices(ballot, weights=voter_weights)
                    remaining_voters = [v for v in ballot.voters if v.name != remove_voter]
                    remaining_ballot = Ballot(voters=remaining_voters, alternatives=ballot.alternatives)
                    remaining_weights = [v.voting_weight for v in remaining_voters]
                    new_power = compute_power_indices(remaining_ballot, weights=remaining_weights)

                # Before table
                st.subheader("修改前权力分布")
                import pandas as pd
                before_data = []
                for name in orig_power.shapley_shubik:
                    before_data.append({
                        "投票人": name,
                        "SS指数": f"{orig_power.shapley_shubik[name]:.4f}",
                        "Banzhaf": f"{orig_power.banzhaf[name]:.4f}",
                    })
                st.dataframe(pd.DataFrame(before_data), width="stretch")

                # After table
                st.subheader(f"移除 {remove_voter} 后")
                after_data = []
                for name in new_power.shapley_shubik:
                    old_ss = orig_power.shapley_shubik.get(name, 0)
                    new_ss = new_power.shapley_shubik[name]
                    delta = new_ss - old_ss
                    after_data.append({
                        "投票人": name,
                        "SS指数": f"{new_ss:.4f}",
                        "变化": f"{delta:+.4f}",
                        "Banzhaf": f"{new_power.banzhaf[name]:.4f}",
                    })
                st.dataframe(pd.DataFrame(after_data), width="stretch")

                # Interpretation
                most_affected = max(
                    new_power.shapley_shubik.keys(),
                    key=lambda n: abs(new_power.shapley_shubik[n] - orig_power.shapley_shubik.get(n, 0))
                )
                delta = new_power.shapley_shubik[most_affected] - orig_power.shapley_shubik.get(most_affected, 0)
                if delta > 0:
                    st.info(f"移除 {remove_voter} 后，**{most_affected}** 的权力增加最多（+{delta:.4f}）。")
                else:
                    st.info(f"移除 {remove_voter} 后，所有剩余投票人的权力均有所增加。")
        else:
            st.info("投票人少于3人时，移除模拟意义有限。")
