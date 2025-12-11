use winterfell::{
    Air, AirContext, Assertion, EvaluationFrame, ProofOptions, TraceInfo,
    TransitionConstraintDegree,
};
use winterfell::math::{fields::f62::BaseElement, FieldElement};

pub struct GdAir {
    context: AirContext<BaseElement>,
    eta: BaseElement,
}

impl Air for GdAir {
    type BaseField = BaseElement;
    type PublicInputs = BaseElement;

    fn new(
        trace_info: TraceInfo,
        pub_inputs: Self::PublicInputs,
        options: ProofOptions,
    ) -> Self {
        let degrees = vec![TransitionConstraintDegree::new(1)];
        let context = AirContext::new(trace_info, degrees, 0, options);
        Self { context, eta: pub_inputs }
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }

    fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let w = frame.current()[0];
        let g = frame.current()[1];
        let wp = frame.next()[0];

        let eta = E::from(self.eta);

        result[0] = w - eta * g - wp;
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        vec![]
    }
}
