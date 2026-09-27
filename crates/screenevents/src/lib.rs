//! ScreenEvents wire types and deterministic structural tracking.

#![forbid(unsafe_code)]
mod compact;
mod model;
mod reducer;
mod tracker;

pub use compact::StatefulCompactRenderer;
pub use model::{
    CloudEvent, ElementContentChangedData, ElementData, ElementDisappearedData, ElementMovedData,
    ElementObservation, ElementState, EventBody, FrameSize, ObservationFrame, ObservedActivityData,
    ObservedActivityKind, Rect, ScreenResizedData, SnapshotData,
};
pub use reducer::{ReducerError, StructuralState};
pub use tracker::{EncoderError, ScreenEventEncoder, TrackerRetentionProfile, TrackerWorkProfile};
