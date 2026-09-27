use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Integer pixel rectangle in the decoded media coordinate space.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    #[must_use]
    pub fn area(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    #[must_use]
    pub fn intersection_over_union(self, other: Self) -> f64 {
        let left = u64::from(self.x.max(other.x));
        let top = u64::from(self.y.max(other.y));
        let right = (u64::from(self.x) + u64::from(self.width))
            .min(u64::from(other.x) + u64::from(other.width));
        let bottom = (u64::from(self.y) + u64::from(self.height))
            .min(u64::from(other.y) + u64::from(other.height));

        if right <= left || bottom <= top {
            return 0.0;
        }

        let intersection = (right - left) * (bottom - top);
        let union = self.area() + other.area() - intersection;
        intersection as f64 / union as f64
    }

    #[must_use]
    pub fn fits_within(self, frame: FrameSize) -> bool {
        self.width > 0
            && self.height > 0
            && u64::from(self.x) + u64::from(self.width) <= u64::from(frame.width)
            && u64::from(self.y) + u64::from(self.height) <= u64::from(frame.height)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrameSize {
    pub width: u32,
    pub height: u32,
}

impl FrameSize {
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.width > 0 && self.height > 0
    }
}

/// One frame's perception observation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ElementObservation {
    #[serde(rename = "box")]
    pub bbox: Rect,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub state: BTreeMap<String, String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationFrame {
    pub timestamp_ms: u64,
    pub width: u32,
    pub height: u32,
    pub elements: Vec<ElementObservation>,
}

impl ObservationFrame {
    #[must_use]
    pub fn size(&self) -> FrameSize {
        FrameSize {
            width: self.width,
            height: self.height,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ElementState {
    pub id: String,
    #[serde(rename = "box")]
    pub bbox: Rect,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub state: BTreeMap<String, String>,
    pub confidence: f32,
}

impl ElementState {
    #[must_use]
    pub fn from_observation(id: String, observation: ElementObservation) -> Self {
        Self {
            id,
            bbox: observation.bbox,
            text: observation.text,
            role: observation.role,
            state: observation.state,
            // Confidence is structural output, so quantize detector noise before
            // deciding whether consumers need a state update.
            confidence: (observation.confidence * 10.0).round() / 10.0,
        }
    }

    #[must_use]
    pub fn semantic_content_eq(&self, other: &Self) -> bool {
        self.text == other.text
            && self.role == other.role
            && self.state == other.state
            && self.confidence == other.confidence
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotData {
    pub frame: FrameSize,
    pub elements: Vec<ElementState>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ElementData {
    pub element: ElementState,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ElementMovedData {
    pub id: String,
    pub from: Rect,
    pub to: Rect,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ElementContentChangedData {
    pub element: ElementState,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ElementDisappearedData {
    pub id: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScreenResizedData {
    pub from: FrameSize,
    pub to: FrameSize,
}

/// One interval-scoped observation of visible activity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservedActivityData {
    pub from_frametime: u64,
    pub kind: ObservedActivityKind,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObservedActivityKind {
    ElementRegionChanged {
        element_id: String,
    },
    RegionTranslated {
        #[serde(rename = "box")]
        bbox: Rect,
        dx: i32,
        dy: i32,
    },
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum StrictObservedActivityData {
    ElementRegionChanged {
        from_frametime: u64,
        element_id: String,
    },
    RegionTranslated {
        from_frametime: u64,
        #[serde(rename = "box")]
        bbox: Rect,
        dx: i32,
        dy: i32,
    },
}

impl Serialize for ObservedActivityData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match &self.kind {
            ObservedActivityKind::ElementRegionChanged { element_id } => {
                StrictObservedActivityData::ElementRegionChanged {
                    from_frametime: self.from_frametime,
                    element_id: element_id.clone(),
                }
                .serialize(serializer)
            }
            ObservedActivityKind::RegionTranslated { bbox, dx, dy } => {
                StrictObservedActivityData::RegionTranslated {
                    from_frametime: self.from_frametime,
                    bbox: *bbox,
                    dx: *dx,
                    dy: *dy,
                }
                .serialize(serializer)
            }
        }
    }
}

impl<'de> Deserialize<'de> for ObservedActivityData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(
            match StrictObservedActivityData::deserialize(deserializer)? {
                StrictObservedActivityData::ElementRegionChanged {
                    from_frametime,
                    element_id,
                } => Self {
                    from_frametime,
                    kind: ObservedActivityKind::ElementRegionChanged { element_id },
                },
                StrictObservedActivityData::RegionTranslated {
                    from_frametime,
                    bbox,
                    dx,
                    dy,
                } => Self {
                    from_frametime,
                    kind: ObservedActivityKind::RegionTranslated { bbox, dx, dy },
                },
            },
        )
    }
}

/// ScreenEvents v0 structural event body.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum EventBody {
    #[serde(rename = "dev.naky.screenevents.snapshot.v0")]
    Snapshot(SnapshotData),
    #[serde(rename = "dev.naky.screenevents.element.appeared.v0")]
    ElementAppeared(ElementData),
    #[serde(rename = "dev.naky.screenevents.element.moved.v0")]
    ElementMoved(ElementMovedData),
    #[serde(rename = "dev.naky.screenevents.element.content_changed.v0")]
    ElementContentChanged(ElementContentChangedData),
    #[serde(rename = "dev.naky.screenevents.element.disappeared.v0")]
    ElementDisappeared(ElementDisappearedData),
    #[serde(rename = "dev.naky.screenevents.screen.resized.v0")]
    ScreenResized(ScreenResizedData),
    #[serde(rename = "dev.naky.screenevents.activity.observed.v0")]
    ObservedActivity(ObservedActivityData),
}

/// CloudEvents-compatible envelope. `sequence` and `frametime` are extension
/// attributes; media time remains an integer to avoid inventing wall-clock time.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CloudEvent {
    pub specversion: String,
    pub id: String,
    pub source: String,
    pub sequence: u64,
    pub frametime: u64,
    pub datacontenttype: String,
    #[serde(flatten)]
    pub body: EventBody,
}

#[cfg(test)]
mod tests {
    use super::{
        CloudEvent, EventBody, FrameSize, ObservedActivityData, ObservedActivityKind, Rect,
    };

    #[test]
    fn extreme_rectangles_do_not_overflow_geometry() {
        let edge = Rect {
            x: u32::MAX,
            y: u32::MAX,
            width: 1,
            height: 1,
        };
        assert!(!edge.fits_within(FrameSize {
            width: u32::MAX,
            height: u32::MAX,
        }));
        assert_eq!(edge.intersection_over_union(edge), 1.0);
    }

    #[test]
    fn observed_activity_serializes_as_one_flattened_strict_family() {
        let event = CloudEvent {
            specversion: "1.0".to_owned(),
            id: "test:1".to_owned(),
            source: "urn:naky:stream:test".to_owned(),
            sequence: 1,
            frametime: 200,
            datacontenttype: "application/json".to_owned(),
            body: EventBody::ObservedActivity(ObservedActivityData {
                from_frametime: 0,
                kind: ObservedActivityKind::ElementRegionChanged {
                    element_id: "e000001".to_owned(),
                },
            }),
        };
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["type"], "dev.naky.screenevents.activity.observed.v0");
        assert_eq!(value["data"]["from_frametime"], 0);
        assert_eq!(value["data"]["kind"], "element_region_changed");
        assert_eq!(value["data"]["element_id"], "e000001");
        assert_eq!(serde_json::from_value::<CloudEvent>(value).unwrap(), event);
    }

    #[test]
    fn observed_activity_rejects_unknown_envelope_data_and_kind_fields() {
        let base = serde_json::json!({
            "specversion": "1.0",
            "id": "test:1",
            "source": "urn:naky:stream:test",
            "sequence": 1,
            "frametime": 200,
            "datacontenttype": "application/json",
            "type": "dev.naky.screenevents.activity.observed.v0",
            "data": {
                "from_frametime": 0,
                "kind": "region_translated",
                "box": {"x": 0, "y": 0, "width": 4, "height": 4},
                "dx": 0,
                "dy": -4
            }
        });
        let mut envelope_extra = base.clone();
        envelope_extra["surprise"] = serde_json::json!(true);
        assert!(serde_json::from_value::<CloudEvent>(envelope_extra).is_err());
        let mut data_extra = base;
        data_extra["data"]["confidence"] = serde_json::json!(0.9);
        assert!(serde_json::from_value::<CloudEvent>(data_extra).is_err());
    }
}
