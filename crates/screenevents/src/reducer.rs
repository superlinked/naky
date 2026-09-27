use std::collections::BTreeMap;

use thiserror::Error;

use crate::model::{ElementState, EventBody, FrameSize};

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ReducerError {
    #[error("cannot apply a delta before a snapshot")]
    MissingSnapshot,
    #[error("element {0} already exists")]
    DuplicateElement(String),
    #[error("element {0} does not exist")]
    UnknownElement(String),
    #[error("activity interval {from}..{to} is not positive")]
    InvalidActivityInterval { from: u64, to: u64 },
    #[error("activity translation box does not fit within the current frame")]
    InvalidActivityBox,
    #[error("activity translation displacement must be nonzero")]
    ZeroActivityTranslation,
}

/// Reconstructed structural screen state at a point in the event stream.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StructuralState {
    pub frame: Option<FrameSize>,
    pub elements: BTreeMap<String, ElementState>,
}

impl StructuralState {
    pub fn apply(&mut self, event: &EventBody, frametime: u64) -> Result<(), ReducerError> {
        match event {
            EventBody::Snapshot(data) => {
                self.frame = Some(data.frame);
                self.elements = data
                    .elements
                    .iter()
                    .cloned()
                    .map(|element| (element.id.clone(), element))
                    .collect();
            }
            EventBody::ElementAppeared(data) => {
                self.require_snapshot()?;
                if self.elements.contains_key(&data.element.id) {
                    return Err(ReducerError::DuplicateElement(data.element.id.clone()));
                }
                self.elements
                    .insert(data.element.id.clone(), data.element.clone());
            }
            EventBody::ElementMoved(data) => {
                self.require_snapshot()?;
                let element = self
                    .elements
                    .get_mut(&data.id)
                    .ok_or_else(|| ReducerError::UnknownElement(data.id.clone()))?;
                element.bbox = data.to;
            }
            EventBody::ElementContentChanged(data) => {
                self.require_snapshot()?;
                if !self.elements.contains_key(&data.element.id) {
                    return Err(ReducerError::UnknownElement(data.element.id.clone()));
                }
                self.elements
                    .insert(data.element.id.clone(), data.element.clone());
            }
            EventBody::ElementDisappeared(data) => {
                self.require_snapshot()?;
                if self.elements.remove(&data.id).is_none() {
                    return Err(ReducerError::UnknownElement(data.id.clone()));
                }
            }
            EventBody::ScreenResized(data) => {
                self.require_snapshot()?;
                self.frame = Some(data.to);
            }
            EventBody::ObservedActivity(data) => {
                let frame = self.require_snapshot()?;
                if data.from_frametime >= frametime {
                    return Err(ReducerError::InvalidActivityInterval {
                        from: data.from_frametime,
                        to: frametime,
                    });
                }
                match &data.kind {
                    crate::ObservedActivityKind::ElementRegionChanged { element_id } => {
                        if !self.elements.contains_key(element_id) {
                            return Err(ReducerError::UnknownElement(element_id.clone()));
                        }
                    }
                    crate::ObservedActivityKind::RegionTranslated { bbox, dx, dy } => {
                        if !bbox.fits_within(frame) {
                            return Err(ReducerError::InvalidActivityBox);
                        }
                        if *dx == 0 && *dy == 0 {
                            return Err(ReducerError::ZeroActivityTranslation);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn require_snapshot(&self) -> Result<FrameSize, ReducerError> {
        self.frame.ok_or(ReducerError::MissingSnapshot)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{ElementObservation, ObservationFrame, Rect, ScreenEventEncoder};

    fn observation(x: u32, y: u32, text: &str) -> ElementObservation {
        ElementObservation {
            bbox: Rect {
                x,
                y,
                width: 100,
                height: 30,
            },
            text: Some(text.to_owned()),
            role: Some("button".to_owned()),
            state: BTreeMap::new(),
            confidence: 1.0,
        }
    }

    fn frame(timestamp_ms: u64, elements: Vec<ElementObservation>) -> ObservationFrame {
        ObservationFrame {
            timestamp_ms,
            width: 800,
            height: 600,
            elements,
        }
    }

    #[test]
    fn reconstructs_state_from_snapshot_and_deltas() {
        let mut encoder = ScreenEventEncoder::new("reduce").unwrap();
        let frames = [
            frame(0, vec![observation(10, 10, "Save")]),
            frame(33, vec![observation(20, 10, "Saved")]),
            frame(66, vec![]),
            frame(99, vec![observation(20, 10, "saved")]),
        ];
        let events: Vec<_> = frames
            .into_iter()
            .flat_map(|frame| encoder.process_frame(frame).unwrap())
            .collect();

        let mut state = StructuralState::default();
        for event in &events {
            state.apply(&event.body, event.frametime).unwrap();
        }

        assert_eq!(
            state.frame,
            Some(FrameSize {
                width: 800,
                height: 600
            })
        );
        assert_eq!(state.elements.len(), 1);
        let element = &state.elements["e000001"];
        assert_eq!(element.text.as_deref(), Some("saved"));
        assert_eq!(element.bbox.x, 20);
        assert_eq!(element.bbox.y, 10);
    }

    #[test]
    fn rejects_delta_before_snapshot() {
        let event = EventBody::ScreenResized(crate::ScreenResizedData {
            from: FrameSize {
                width: 1,
                height: 1,
            },
            to: FrameSize {
                width: 2,
                height: 2,
            },
        });
        assert_eq!(
            StructuralState::default().apply(&event, 0),
            Err(ReducerError::MissingSnapshot)
        );
    }

    #[test]
    fn validates_activity_without_retaining_activity_history() {
        let mut state = StructuralState::default();
        state
            .apply(
                &EventBody::Snapshot(crate::SnapshotData {
                    frame: FrameSize {
                        width: 800,
                        height: 600,
                    },
                    elements: vec![crate::ElementState::from_observation(
                        "e000001".to_owned(),
                        observation(10, 10, "Save"),
                    )],
                }),
                0,
            )
            .unwrap();
        let before = state.clone();
        state
            .apply(
                &EventBody::ObservedActivity(crate::ObservedActivityData {
                    from_frametime: 0,
                    kind: crate::ObservedActivityKind::ElementRegionChanged {
                        element_id: "e000001".to_owned(),
                    },
                }),
                200,
            )
            .unwrap();
        assert_eq!(state, before);

        let unknown = EventBody::ObservedActivity(crate::ObservedActivityData {
            from_frametime: 0,
            kind: crate::ObservedActivityKind::ElementRegionChanged {
                element_id: "e000002".to_owned(),
            },
        });
        assert_eq!(
            state.apply(&unknown, 200),
            Err(ReducerError::UnknownElement("e000002".to_owned()))
        );
        let zero = EventBody::ObservedActivity(crate::ObservedActivityData {
            from_frametime: 0,
            kind: crate::ObservedActivityKind::RegionTranslated {
                bbox: Rect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                dx: 0,
                dy: 0,
            },
        });
        assert_eq!(
            state.apply(&zero, 200),
            Err(ReducerError::ZeroActivityTranslation)
        );

        let invalid_box = EventBody::ObservedActivity(crate::ObservedActivityData {
            from_frametime: 0,
            kind: crate::ObservedActivityKind::RegionTranslated {
                bbox: Rect {
                    x: 799,
                    y: 0,
                    width: 2,
                    height: 4,
                },
                dx: 1,
                dy: 0,
            },
        });
        assert_eq!(
            state.apply(&invalid_box, 200),
            Err(ReducerError::InvalidActivityBox)
        );

        let invalid_interval = EventBody::ObservedActivity(crate::ObservedActivityData {
            from_frametime: 200,
            kind: crate::ObservedActivityKind::ElementRegionChanged {
                element_id: "e000001".to_owned(),
            },
        });
        assert_eq!(
            state.apply(&invalid_interval, 200),
            Err(ReducerError::InvalidActivityInterval { from: 200, to: 200 })
        );
    }

    #[test]
    fn rejects_activity_before_snapshot() {
        let event = EventBody::ObservedActivity(crate::ObservedActivityData {
            from_frametime: 0,
            kind: crate::ObservedActivityKind::RegionTranslated {
                bbox: Rect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                dx: 1,
                dy: 0,
            },
        });
        assert_eq!(
            StructuralState::default().apply(&event, 200),
            Err(ReducerError::MissingSnapshot)
        );
    }
}
