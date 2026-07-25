from __future__ import annotations

from agent.execution_models import ExecutionPlan, PlanStep
from agent.query_schema import QueryIntent, QueryRequest


class AgentPlanner:
    def plan(self, query_request: QueryRequest) -> ExecutionPlan:
        intent = query_request.intent

        if intent == QueryIntent.broad_analysis:
            return self._plan_broad_analysis(query_request)
        if intent == QueryIntent.structuring_detection:
            return self._plan_structuring(query_request)
        if intent == QueryIntent.threshold_analysis:
            return self._plan_threshold(query_request)
        if intent == QueryIntent.customer_investigation:
            return self._plan_customer_investigation(query_request)
        if intent == QueryIntent.risk_explanation:
            return self._plan_risk_explanation(query_request)
        if intent == QueryIntent.high_risk_search:
            return self._plan_high_risk_search(query_request)
        if intent == QueryIntent.velocity_analysis:
            return self._plan_velocity(query_request)
        if intent == QueryIntent.fan_out_detection:
            return self._plan_fan_out(query_request)
        if intent == QueryIntent.fan_in_detection:
            return self._plan_fan_in(query_request)
        if intent == QueryIntent.cycle_detection:
            return self._plan_cycle(query_request)
        if intent == QueryIntent.gather_scatter_detection:
            return self._plan_gather_scatter(query_request)
        if intent == QueryIntent.scatter_gather_detection:
            return self._plan_scatter_gather(query_request)
        if intent == QueryIntent.smurfing_detection:
            return self._plan_smurfing(query_request)
        if intent == QueryIntent.suspicious_activity_search:
            return self._plan_suspicious_activity(query_request)
        if intent == QueryIntent.eda_request:
            return self._plan_eda(query_request)
        return self._plan_default(query_request)

    def _plan_broad_analysis(self, query_request: QueryRequest) -> ExecutionPlan:
        steps = [
            self._step(1, "dataset_profiler", "Broad analysis begins with dataset profiling.", parameters={"scope": "entire_dataset"}),
            self._step(2, "eda", "Broad analysis needs exploratory context.", parameters={"scope": "entire_dataset"}),
            self._step(3, "feature_engineering", "Broad analysis benefits from AML feature creation.", parameters={"scope": "entire_dataset"}),
            self._step(4, "structuring_detector", "Structuring is a core suspicious activity pattern.", parameters={"scope": "entire_dataset"}),
            self._step(5, "smurfing_detector", "Smurfing is another relevant AML pattern for broad review.", parameters={"scope": "entire_dataset"}),
            self._step(6, "velocity_detector", "Velocity signals help detect burst-like activity.", parameters={"scope": "entire_dataset"}),
            self._step(7, "behavior_deviation_detector", "Behavior drift is relevant for broad AML analysis.", parameters={"scope": "entire_dataset"}),
            self._step(8, "anomaly_detector", "Anomaly scoring adds a complementary risk signal.", parameters={"scope": "entire_dataset"}),
            self._step(9, "risk_scoring", "Signals need a consolidated risk score.", parameters={"scope": "entire_dataset"}),
            self._step(10, "explanation", "Judges need evidence-backed explanations.", parameters={"scope": "entire_dataset"}),
        ]
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=steps,
            skipped_tools=["transaction_filter", "aggregation", "threshold_rule", "customer_lookup", "risk_lookup", "result_filtering"],
            planning_summary="Broad suspicious-activity review; profile the dataset, generate features, run the AML detectors, then score and explain results.",
        )

    def _plan_structuring(self, query_request: QueryRequest) -> ExecutionPlan:
        steps = [
            self._step(1, "transaction_filter", "The query includes a date window and needs scoped transactions.", parameters=self._filter_parameters(query_request)),
            self._step(2, "feature_engineering", "Structuring relies on frequency and near-threshold features.", parameters=self._scope_parameters(query_request)),
            self._step(3, "structuring_detector", "The user explicitly requested structuring detection.", parameters=self._scope_parameters(query_request)),
            self._step(4, "risk_scoring", "Detected signals should be converted to an interpretable risk score.", parameters=self._scope_parameters(query_request)),
            self._step(5, "explanation", "The result should include concise evidence and rationale.", parameters=self._scope_parameters(query_request)),
        ]
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=steps,
            skipped_tools=["dataset_profiler", "eda", "smurfing_detector", "velocity_detector", "behavior_deviation_detector", "anomaly_detector"],
            planning_summary="Targeted structuring query; use scoped filtering and the structuring-specific path only.",
        )

    def _plan_threshold(self, query_request: QueryRequest) -> ExecutionPlan:
        steps = [
            self._step(1, "transaction_filter", "The query has direct count and amount constraints.", parameters=self._filter_parameters(query_request)),
            self._step(2, "aggregation", "The query asks for customers meeting a transaction-count condition.", parameters=self._scope_parameters(query_request)),
            self._step(3, "threshold_rule", "A deterministic threshold rule is sufficient.", parameters=self._threshold_parameters(query_request)),
        ]
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=steps,
            skipped_tools=["dataset_profiler", "eda", "feature_engineering", "structuring_detector", "smurfing_detector", "velocity_detector", "behavior_deviation_detector", "anomaly_detector", "risk_scoring", "explanation"],
            planning_summary="Threshold query; direct filtering, aggregation, and rule evaluation are enough, so no ML detectors are needed.",
        )

    def _plan_customer_investigation(self, query_request: QueryRequest) -> ExecutionPlan:
        steps = [
            self._step(1, "customer_lookup", "The query targets a single customer.", parameters=self._customer_parameters(query_request)),
            self._step(2, "feature_engineering", "A customer case file should include customer-scoped features.", parameters=self._customer_parameters(query_request)),
            self._step(3, "behavior_deviation_detector", "Single-customer investigation should check behavior drift.", parameters=self._customer_parameters(query_request)),
            self._step(4, "anomaly_detector", "Anomaly evidence can complement the customer review.", parameters=self._customer_parameters(query_request)),
            self._step(5, "risk_scoring", "The customer needs a consolidated risk score.", parameters=self._customer_parameters(query_request)),
            self._step(6, "explanation", "The case file must explain why the customer is suspicious.", parameters=self._customer_parameters(query_request)),
        ]
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=steps,
            skipped_tools=["dataset_profiler", "eda", "transaction_filter", "aggregation", "threshold_rule", "smurfing_detector", "velocity_detector"],
            planning_summary="Single-customer investigation; stay customer-scoped and avoid full-dataset EDA.",
        )

    def _plan_risk_explanation(self, query_request: QueryRequest) -> ExecutionPlan:
        steps = [
            self._step(1, "customer_lookup", "The query is about one customer's risk status.", parameters=self._customer_parameters(query_request)),
            self._step(2, "risk_lookup", "Retrieve existing risk evidence before explaining it.", parameters=self._customer_parameters(query_request)),
            self._step(3, "explanation", "The user wants the reason for the high-risk label.", parameters=self._customer_parameters(query_request)),
        ]
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=steps,
            skipped_tools=["dataset_profiler", "eda", "feature_engineering", "transaction_filter", "aggregation", "threshold_rule", "smurfing_detector", "velocity_detector", "behavior_deviation_detector", "anomaly_detector", "risk_scoring"],
            planning_summary="Risk explanation request; retrieve customer evidence and explain it without rerunning broad analysis.",
        )

    def _plan_high_risk_search(self, query_request: QueryRequest) -> ExecutionPlan:
        steps = [
            self._step(1, "feature_engineering", "High-risk search can reuse customer-level feature summaries.", parameters={"scope": "high_risk_search"}),
            self._step(2, "risk_lookup", "Query existing risk records or evidence for filtering.", parameters={"scope": "high_risk_search"}),
            self._step(3, "result_filtering", "Filter the results to the requested high-risk subset.", parameters={"scope": "high_risk_search"}),
        ]
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=steps,
            skipped_tools=["dataset_profiler", "eda", "transaction_filter", "aggregation", "threshold_rule", "structuring_detector", "smurfing_detector", "velocity_detector", "behavior_deviation_detector", "anomaly_detector", "explanation"],
            planning_summary="High-risk search should filter existing evidence rather than rerun the full detector stack.",
        )

    def _plan_velocity(self, query_request: QueryRequest) -> ExecutionPlan:
        steps = [
            self._step(1, "feature_engineering", "Velocity detection needs temporal transaction features.", parameters=self._scope_parameters(query_request)),
            self._step(2, "velocity_detector", "The user explicitly requested unusual transaction velocity.", parameters=self._scope_parameters(query_request)),
            self._step(3, "risk_scoring", "Velocity evidence should be turned into a clear risk score.", parameters=self._scope_parameters(query_request)),
            self._step(4, "explanation", "The result should explain the velocity evidence.", parameters=self._scope_parameters(query_request)),
        ]
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=steps,
            skipped_tools=["dataset_profiler", "eda", "transaction_filter", "aggregation", "threshold_rule", "customer_lookup", "structuring_detector", "smurfing_detector", "behavior_deviation_detector", "anomaly_detector", "risk_lookup"],
            planning_summary="Velocity-focused query; use temporal features and the velocity detector only.",
        )

    def _plan_fan_out(self, query_request: QueryRequest) -> ExecutionPlan:
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=[
                self._step(1, "fan_out_detector", "The user requested fan-out behavior: one source sending to many distinct receivers.", parameters=self._scope_parameters(query_request)),
            ],
            skipped_tools=["dataset_profiler", "eda", "transaction_filter", "aggregation", "threshold_rule", "customer_lookup", "feature_engineering", "structuring_detector", "smurfing_detector", "fan_in_detector", "velocity_detector", "behavior_deviation_detector", "anomaly_detector", "risk_scoring", "explanation", "risk_lookup"],
            planning_summary="Fan-out query; run only the targeted fan-out detector.",
        )

    def _plan_fan_in(self, query_request: QueryRequest) -> ExecutionPlan:
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=[
                self._step(1, "fan_in_detector", "The user requested fan-in behavior: one destination receiving from many distinct senders.", parameters=self._scope_parameters(query_request)),
            ],
            skipped_tools=["dataset_profiler", "eda", "transaction_filter", "aggregation", "threshold_rule", "customer_lookup", "feature_engineering", "structuring_detector", "smurfing_detector", "fan_out_detector", "velocity_detector", "behavior_deviation_detector", "anomaly_detector", "risk_scoring", "explanation", "risk_lookup"],
            planning_summary="Fan-in query; run only the targeted fan-in detector.",
        )

    def _plan_cycle(self, query_request: QueryRequest) -> ExecutionPlan:
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=[
                self._step(1, "cycle_detector", "The user requested a circular transaction cycle.", parameters=self._scope_parameters(query_request)),
            ],
            skipped_tools=["dataset_profiler", "eda", "transaction_filter", "aggregation", "threshold_rule", "customer_lookup", "feature_engineering", "structuring_detector", "smurfing_detector", "fan_out_detector", "fan_in_detector", "velocity_detector", "behavior_deviation_detector", "anomaly_detector", "risk_scoring", "explanation", "risk_lookup"],
            planning_summary="Cycle query; run only the targeted cycle detector.",
        )

    def _plan_gather_scatter(self, query_request: QueryRequest) -> ExecutionPlan:
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=[
                self._step(1, "gather_scatter_detector", "The user requested gather-scatter behavior.", parameters=self._scope_parameters(query_request)),
            ],
            skipped_tools=["dataset_profiler", "eda", "transaction_filter", "aggregation", "threshold_rule", "customer_lookup", "feature_engineering", "structuring_detector", "smurfing_detector", "fan_out_detector", "fan_in_detector", "velocity_detector", "behavior_deviation_detector", "anomaly_detector", "risk_scoring", "explanation", "risk_lookup"],
            planning_summary="Gather-scatter query; run only the targeted gather-scatter detector.",
        )

    def _plan_scatter_gather(self, query_request: QueryRequest) -> ExecutionPlan:
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=[
                self._step(1, "scatter_gather_detector", "The user requested scatter-gather behavior.", parameters=self._scope_parameters(query_request)),
            ],
            skipped_tools=["dataset_profiler", "eda", "transaction_filter", "aggregation", "threshold_rule", "customer_lookup", "feature_engineering", "structuring_detector", "smurfing_detector", "fan_out_detector", "fan_in_detector", "velocity_detector", "behavior_deviation_detector", "anomaly_detector", "risk_scoring", "explanation", "risk_lookup"],
            planning_summary="Scatter-gather query; run only the targeted scatter-gather detector.",
        )

    def _plan_smurfing(self, query_request: QueryRequest) -> ExecutionPlan:
        steps = [
            self._step(1, "feature_engineering", "Smurfing detection needs counterparties and repeated-amount features.", parameters=self._scope_parameters(query_request)),
            self._step(2, "smurfing_detector", "The query explicitly asks for smurfing patterns.", parameters=self._scope_parameters(query_request)),
            self._step(3, "risk_scoring", "Smurfing evidence should be scored consistently.", parameters=self._scope_parameters(query_request)),
            self._step(4, "explanation", "The result should state why smurfing is suspected.", parameters=self._scope_parameters(query_request)),
        ]
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=steps,
            skipped_tools=["dataset_profiler", "eda", "transaction_filter", "aggregation", "threshold_rule", "customer_lookup", "structuring_detector", "velocity_detector", "behavior_deviation_detector", "anomaly_detector"],
            planning_summary="Smurfing-focused query; select the counterparty-heavy path and skip unrelated detectors.",
        )

    def _plan_suspicious_activity(self, query_request: QueryRequest) -> ExecutionPlan:
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=[
                self._step(1, "dataset_profiler", "A broad suspicious-activity request starts with dataset profiling.", parameters={"scope": "entire_dataset"}),
                self._step(2, "eda", "Broad requests benefit from general exploration.", parameters={"scope": "entire_dataset"}),
                self._step(3, "feature_engineering", "General suspicious-activity analysis needs baseline features.", parameters={"scope": "entire_dataset"}),
                self._step(4, "structuring_detector", "Structuring is a primary suspicious pattern.", parameters={"scope": "entire_dataset"}),
                self._step(5, "smurfing_detector", "Smurfing is also relevant for a broad investigation.", parameters={"scope": "entire_dataset"}),
                self._step(6, "velocity_detector", "Velocity anomalies help detect suspicious bursts.", parameters={"scope": "entire_dataset"}),
                self._step(7, "behavior_deviation_detector", "Behavior deviation helps identify unusual customers.", parameters={"scope": "entire_dataset"}),
                self._step(8, "anomaly_detector", "Anomaly scoring adds a complementary signal.", parameters={"scope": "entire_dataset"}),
                self._step(9, "risk_scoring", "Multiple signals should be combined into a risk score.", parameters={"scope": "entire_dataset"}),
                self._step(10, "explanation", "The final output must explain the flags.", parameters={"scope": "entire_dataset"}),
            ],
            skipped_tools=["transaction_filter", "aggregation", "threshold_rule", "customer_lookup", "risk_lookup", "result_filtering"],
            planning_summary="Broad suspicious-activity query; inspect the dataset, run the broad detector set, then score and explain.",
        )

    def _plan_eda(self, query_request: QueryRequest) -> ExecutionPlan:
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=[
                self._step(1, "dataset_profiler", "The query requests exploratory analysis.", parameters={"scope": "entire_dataset"}),
                self._step(2, "eda", "EDA is the direct response to the user's request.", parameters={"scope": "entire_dataset"}),
            ],
            skipped_tools=["feature_engineering", "transaction_filter", "aggregation", "threshold_rule", "structuring_detector", "smurfing_detector", "velocity_detector", "behavior_deviation_detector", "anomaly_detector", "risk_scoring", "explanation"],
            planning_summary="EDA request; only profiling and exploration are needed.",
        )

    def _plan_default(self, query_request: QueryRequest) -> ExecutionPlan:
        return ExecutionPlan(
            query_intent=query_request.intent,
            steps=[self._step(1, "dataset_profiler", "Default to a safe dataset overview when intent is ambiguous.", parameters={"scope": "entire_dataset"})],
            skipped_tools=["eda", "transaction_filter", "aggregation", "threshold_rule", "customer_lookup", "feature_engineering", "structuring_detector", "smurfing_detector", "velocity_detector", "behavior_deviation_detector", "anomaly_detector", "risk_scoring", "explanation", "risk_lookup", "result_filtering"],
            planning_summary="Fallback plan for ambiguous input; gather a basic dataset overview only.",
        )

    def _step(
        self,
        order: int,
        tool_name: str,
        reason: str,
        *,
        required: bool = True,
        parameters: dict[str, object] | None = None,
        dependencies: list[str] | None = None,
    ) -> PlanStep:
        return PlanStep(
            order=order,
            tool_name=tool_name,
            reason=reason,
            required=required,
            parameters=parameters or {},
            dependencies=dependencies or [],
        )

    def _scope_parameters(self, query_request: QueryRequest) -> dict[str, object]:
        parameters: dict[str, object] = {}
        if query_request.customer_id:
            parameters["customer_id"] = query_request.customer_id
        if query_request.date_range:
            parameters["date_range"] = query_request.date_range.model_dump(mode="json")
        if query_request.amount_threshold:
            parameters["amount_threshold"] = query_request.amount_threshold.model_dump(mode="json")
        if query_request.transaction_count_threshold:
            parameters["transaction_count_threshold"] = query_request.transaction_count_threshold.model_dump(mode="json")
        if query_request.transaction_type:
            parameters["transaction_type"] = query_request.transaction_type
        if query_request.country:
            parameters["country"] = query_request.country
        if query_request.aml_pattern:
            parameters["pattern"] = query_request.aml_pattern.value
        return parameters

    def _filter_parameters(self, query_request: QueryRequest) -> dict[str, object]:
        parameters = self._scope_parameters(query_request)
        if query_request.date_range and query_request.date_range.relative_days is not None:
            parameters["relative_days"] = query_request.date_range.relative_days
        return parameters

    def _threshold_parameters(self, query_request: QueryRequest) -> dict[str, object]:
        parameters = self._scope_parameters(query_request)
        return parameters

    def _customer_parameters(self, query_request: QueryRequest) -> dict[str, object]:
        return self._scope_parameters(query_request)
