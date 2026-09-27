use std::collections::BTreeMap;
use std::io::{self, Write};

use thiserror::Error;

use crate::{CloudEvent, EventBody, StructuralState};

const MEANINGFUL_MOVE_PX: u32 = 16;
const MAX_PENDING_FACTS: usize = 3_000;

#[derive(Debug, Error)]
pub enum CompactRenderError {
    #[error("event sequence is not contiguous: expected {expected}, found {actual}")]
    Sequence { expected: u64, actual: u64 },
    #[error("event stream contains multiple sources")]
    Source,
    #[error("invalid structural event stream: {0}")]
    State(#[from] crate::ReducerError),
    #[error("event sequence cannot continue after {0}")]
    SequenceOverflow(u64),
    #[error("invalid canonical element ID {0:?}")]
    ElementId(String),
    #[error("failed to write compact projection: {0}")]
    Io(#[from] io::Error),
}

#[derive(Debug)]
struct ProjectedElement {
    position: (u32, u32),
    text: String,
}

#[derive(Debug)]
struct PendingTextRow {
    id: u64,
    position: Option<(u32, u32)>,
    text: Option<String>,
}

#[derive(Debug)]
enum PendingRun {
    Text {
        frametime: u64,
        operation: char,
        rows: Vec<PendingTextRow>,
    },
    Removal {
        frametime: u64,
        ids: Vec<u64>,
    },
    ElementActivity {
        frametime: u64,
        duration: u64,
        ids: Vec<u64>,
    },
}

impl PendingRun {
    fn len(&self) -> usize {
        match self {
            Self::Text { rows, .. } => rows.len(),
            Self::Removal { ids, .. } | Self::ElementActivity { ids, .. } => ids.len(),
        }
    }

    fn frametime(&self) -> u64 {
        match self {
            Self::Text { frametime, .. }
            | Self::Removal { frametime, .. }
            | Self::ElementActivity { frametime, .. } => *frametime,
        }
    }
}

#[derive(Debug)]
enum RenderedFact {
    Text {
        operation: char,
        row: PendingTextRow,
    },
    Removal(u64),
    ElementActivity {
        duration: u64,
        id: u64,
    },
    Translation {
        duration: u64,
        bbox: crate::Rect,
        dx: i32,
        dy: i32,
    },
    Screen {
        operation: char,
        width: u32,
        height: u32,
    },
}

#[derive(Clone, Copy, Debug)]
enum PendingKey {
    Text(char),
    Removal,
    ElementActivity(u64),
    Immediate,
}

/// Incrementally render canonical ScreenEvents as a stateful compact prompt view.
///
/// References are derived directly from canonical numeric element IDs. The only
/// retained projection state belongs to elements on the current screen.
pub struct StatefulCompactRenderer<W: Write> {
    writer: W,
    source: Option<String>,
    expected_sequence: Option<u64>,
    state: StructuralState,
    projected: BTreeMap<u64, ProjectedElement>,
    current_time: Option<u64>,
    time_header_written: bool,
    pending: Option<PendingRun>,
    started: bool,
}

impl<W: Write> StatefulCompactRenderer<W> {
    #[must_use]
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            source: None,
            expected_sequence: None,
            state: StructuralState::default(),
            projected: BTreeMap::new(),
            current_time: None,
            time_header_written: false,
            pending: None,
            started: false,
        }
    }

    pub fn write_event(&mut self, event: &CloudEvent) -> Result<(), CompactRenderError> {
        if let Err(error) = self.validate_envelope(event) {
            self.flush_pending()?;
            return Err(error);
        }
        if let Err(error) = validate_body_ids(&event.body) {
            self.flush_pending()?;
            return Err(error);
        }

        let content_text_changed = match &event.body {
            EventBody::ElementContentChanged(data) => {
                self.state
                    .elements
                    .get(&data.element.id)
                    .and_then(|element| element.text.as_deref())
                    != data.element.text.as_deref()
            }
            _ => false,
        };
        let disappeared_had_text = match &event.body {
            EventBody::ElementDisappeared(data) => self
                .state
                .elements
                .get(&data.id)
                .is_some_and(|element| element.text.is_some()),
            _ => false,
        };
        let moved_text = match &event.body {
            EventBody::ElementMoved(data)
                if meaningful_move(data.from.x, data.from.y, data.to.x, data.to.y) =>
            {
                Some(
                    self.state
                        .elements
                        .get(&data.id)
                        .and_then(|element| element.text.clone())
                        .unwrap_or_else(|| data.id.clone()),
                )
            }
            _ => None,
        };

        let pending_key = match &event.body {
            EventBody::Snapshot(_) | EventBody::ScreenResized(_) => Some(PendingKey::Immediate),
            EventBody::ElementAppeared(data) => {
                data.element.text.as_ref().map(|_| PendingKey::Text('+'))
            }
            EventBody::ElementContentChanged(data) => (content_text_changed
                && data.element.text.is_some())
            .then_some(PendingKey::Text('~')),
            EventBody::ElementDisappeared(_) => disappeared_had_text.then_some(PendingKey::Removal),
            EventBody::ElementMoved(_) => moved_text.as_ref().map(|_| PendingKey::Text('>')),
            EventBody::ObservedActivity(data) => match &data.kind {
                crate::ObservedActivityKind::ElementRegionChanged { .. } => event
                    .frametime
                    .checked_sub(data.from_frametime)
                    .map(PendingKey::ElementActivity)
                    .or(Some(PendingKey::Immediate)),
                crate::ObservedActivityKind::RegionTranslated { .. } => Some(PendingKey::Immediate),
            },
        };
        self.flush_before_apply(event.frametime, pending_key)?;

        if let Err(error) = self.state.apply(&event.body, event.frametime) {
            self.flush_pending()?;
            return Err(error.into());
        }
        if !self.started {
            self.writer.write_all(b"screenevents-state\n")?;
            self.started = true;
        }
        if self.current_time != Some(event.frametime) {
            self.current_time = Some(event.frametime);
            self.time_header_written = false;
        }

        let fact = match &event.body {
            EventBody::Snapshot(snapshot) => {
                self.projected.clear();
                self.write_immediate(RenderedFact::Screen {
                    operation: 'S',
                    width: snapshot.frame.width,
                    height: snapshot.frame.height,
                })?;
                for element in &snapshot.elements {
                    if let Some(text) = element.text.as_deref() {
                        let id = projection_number(&element.id)?;
                        self.projected.insert(
                            id,
                            ProjectedElement {
                                position: (element.bbox.x, element.bbox.y),
                                text: text.to_owned(),
                            },
                        );
                        self.queue_fact(RenderedFact::Text {
                            operation: '=',
                            row: PendingTextRow {
                                id,
                                position: Some((element.bbox.x, element.bbox.y)),
                                text: Some(text.to_owned()),
                            },
                        })?;
                    }
                }
                return Ok(());
            }
            EventBody::ElementAppeared(data) => {
                if let Some(text) = data.element.text.as_deref() {
                    let id = projection_number(&data.element.id)?;
                    self.projected.insert(
                        id,
                        ProjectedElement {
                            position: (data.element.bbox.x, data.element.bbox.y),
                            text: text.to_owned(),
                        },
                    );
                    Some(RenderedFact::Text {
                        operation: '+',
                        row: PendingTextRow {
                            id,
                            position: Some((data.element.bbox.x, data.element.bbox.y)),
                            text: Some(text.to_owned()),
                        },
                    })
                } else {
                    None
                }
            }
            EventBody::ElementContentChanged(data) => {
                if content_text_changed && let Some(text) = data.element.text.as_deref() {
                    let id = projection_number(&data.element.id)?;
                    let position = (data.element.bbox.x, data.element.bbox.y);
                    let rendered_position = (self.projected.get(&id).map(|item| item.position)
                        != Some(position))
                    .then_some(position);
                    self.projected.insert(
                        id,
                        ProjectedElement {
                            position,
                            text: text.to_owned(),
                        },
                    );
                    Some(RenderedFact::Text {
                        operation: '~',
                        row: PendingTextRow {
                            id,
                            position: rendered_position,
                            text: Some(text.to_owned()),
                        },
                    })
                } else {
                    None
                }
            }
            EventBody::ElementDisappeared(data) => {
                let id = projection_number(&data.id)?;
                self.projected.remove(&id);
                disappeared_had_text.then_some(RenderedFact::Removal(id))
            }
            EventBody::ElementMoved(data) => {
                if let Some(text) = moved_text {
                    let id = projection_number(&data.id)?;
                    let include_text = self.projected.get(&id).is_none_or(|item| item.text != text);
                    self.projected.insert(
                        id,
                        ProjectedElement {
                            position: (data.to.x, data.to.y),
                            text,
                        },
                    );
                    Some(RenderedFact::Text {
                        operation: '>',
                        row: PendingTextRow {
                            id,
                            position: Some((data.to.x, data.to.y)),
                            text: include_text.then(|| {
                                self.projected
                                    .get(&id)
                                    .expect("projected move was inserted")
                                    .text
                                    .clone()
                            }),
                        },
                    })
                } else {
                    None
                }
            }
            EventBody::ScreenResized(data) => Some(RenderedFact::Screen {
                operation: 'R',
                width: data.to.width,
                height: data.to.height,
            }),
            EventBody::ObservedActivity(data) => {
                let duration = event.frametime - data.from_frametime;
                match &data.kind {
                    crate::ObservedActivityKind::ElementRegionChanged { element_id } => {
                        let id = projection_number(element_id)?;
                        Some(RenderedFact::ElementActivity { duration, id })
                    }
                    crate::ObservedActivityKind::RegionTranslated { bbox, dx, dy } => {
                        Some(RenderedFact::Translation {
                            duration,
                            bbox: *bbox,
                            dx: *dx,
                            dy: *dy,
                        })
                    }
                }
            }
        };
        if let Some(fact) = fact {
            self.queue_fact(fact)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<W, CompactRenderError> {
        self.flush_pending()?;
        self.writer.flush()?;
        Ok(self.writer)
    }

    fn flush_before_apply(
        &mut self,
        frametime: u64,
        pending_key: Option<PendingKey>,
    ) -> Result<(), CompactRenderError> {
        let Some(pending) = self.pending.as_ref() else {
            return Ok(());
        };
        let compatible = if pending.frametime() != frametime {
            false
        } else {
            match (pending, pending_key) {
                (PendingRun::Text { operation, .. }, Some(PendingKey::Text(next_operation))) => {
                    *operation == next_operation
                }
                (PendingRun::Removal { .. }, Some(PendingKey::Removal)) => true,
                (
                    PendingRun::ElementActivity { duration, .. },
                    Some(PendingKey::ElementActivity(next_duration)),
                ) => *duration == next_duration,
                (_, None) => true,
                (_, Some(PendingKey::Immediate)) | (_, Some(_)) => false,
            }
        };
        if !compatible || pending.len() == MAX_PENDING_FACTS {
            self.flush_pending()?;
        }
        Ok(())
    }

    fn queue_fact(&mut self, fact: RenderedFact) -> Result<(), CompactRenderError> {
        if matches!(
            fact,
            RenderedFact::Screen { .. } | RenderedFact::Translation { .. }
        ) {
            return self.write_immediate(fact);
        }

        let compatible = match (&self.pending, &fact) {
            (
                Some(PendingRun::Text {
                    frametime,
                    operation,
                    ..
                }),
                RenderedFact::Text {
                    operation: next_operation,
                    ..
                },
            ) => *frametime == self.current_frametime() && operation == next_operation,
            (Some(PendingRun::Removal { frametime, .. }), RenderedFact::Removal(_)) => {
                *frametime == self.current_frametime()
            }
            (
                Some(PendingRun::ElementActivity {
                    frametime,
                    duration,
                    ..
                }),
                RenderedFact::ElementActivity {
                    duration: next_duration,
                    ..
                },
            ) => *frametime == self.current_frametime() && duration == next_duration,
            _ => false,
        };
        if !compatible
            || self
                .pending
                .as_ref()
                .is_some_and(|run| run.len() == MAX_PENDING_FACTS)
        {
            self.flush_pending()?;
        }

        let frametime = self.current_frametime();
        match fact {
            RenderedFact::Text { operation, row } => {
                if let Some(PendingRun::Text { rows, .. }) = self.pending.as_mut() {
                    rows.push(row);
                } else {
                    self.pending = Some(PendingRun::Text {
                        frametime,
                        operation,
                        rows: vec![row],
                    });
                }
            }
            RenderedFact::Removal(id) => {
                if let Some(PendingRun::Removal { ids, .. }) = self.pending.as_mut() {
                    ids.push(id);
                } else {
                    self.pending = Some(PendingRun::Removal {
                        frametime,
                        ids: vec![id],
                    });
                }
            }
            RenderedFact::ElementActivity { duration, id } => {
                if let Some(PendingRun::ElementActivity { ids, .. }) = self.pending.as_mut() {
                    ids.push(id);
                } else {
                    self.pending = Some(PendingRun::ElementActivity {
                        frametime,
                        duration,
                        ids: vec![id],
                    });
                }
            }
            RenderedFact::Screen { .. } | RenderedFact::Translation { .. } => unreachable!(),
        }
        Ok(())
    }

    fn write_immediate(&mut self, fact: RenderedFact) -> Result<(), CompactRenderError> {
        self.flush_pending()?;
        self.write_fact_header()?;
        match fact {
            RenderedFact::Screen {
                operation,
                width,
                height,
            } => writeln!(self.writer, "{operation}{width},{height}")?,
            RenderedFact::Translation {
                duration,
                bbox,
                dx,
                dy,
            } => writeln!(
                self.writer,
                "^{duration} {} {} {} {} {dx} {dy}",
                bbox.x, bbox.y, bbox.width, bbox.height
            )?,
            _ => unreachable!(),
        }
        Ok(())
    }

    fn flush_pending(&mut self) -> Result<(), CompactRenderError> {
        let Some(pending) = self.pending.take() else {
            return Ok(());
        };
        debug_assert!(pending.len() > 0);
        debug_assert_eq!(self.current_time, Some(pending.frametime()));
        match pending {
            PendingRun::Text {
                operation, rows, ..
            } => {
                self.write_fact_header()?;
                self.write_text_rows(operation, rows)?;
            }
            PendingRun::Removal { ids, .. } => {
                self.write_fact_header()?;
                write!(self.writer, "-")?;
                self.write_id_ranges(&ids)?;
                writeln!(self.writer)?;
            }
            PendingRun::ElementActivity { duration, ids, .. } => {
                if !self.time_header_written {
                    write!(
                        self.writer,
                        "@{} ",
                        self.current_time
                            .expect("an event time is set before facts")
                    )?;
                    self.time_header_written = true;
                } else {
                    self.write_fact_header()?;
                }
                if duration == 200 {
                    write!(self.writer, "!")?;
                } else {
                    write!(self.writer, "!{duration}/")?;
                }
                self.write_id_ranges(&ids)?;
                writeln!(self.writer)?;
            }
        }
        Ok(())
    }

    fn write_text_rows(
        &mut self,
        operation: char,
        rows: Vec<PendingTextRow>,
    ) -> Result<(), CompactRenderError> {
        let mut previous_id = None;
        for (index, row) in rows.into_iter().enumerate() {
            if index == 0 {
                write!(self.writer, "{operation}")?;
            }
            let omit_id =
                index > 0 && previous_id.and_then(|id: u64| id.checked_add(1)) == Some(row.id);
            if !omit_id {
                write!(self.writer, "{}", row.id)?;
            }
            if let Some((x, y)) = row.position {
                if !omit_id {
                    write!(self.writer, " ")?;
                }
                write!(self.writer, "{x},{y}")?;
                if let Some(text) = row.text.as_deref() {
                    write!(self.writer, " {}", state_text(text))?;
                }
            } else if let Some(text) = row.text.as_deref() {
                write!(self.writer, ":{}", state_text(text))?;
            }
            writeln!(self.writer)?;
            previous_id = Some(row.id);
        }
        Ok(())
    }

    fn write_id_ranges(&mut self, ids: &[u64]) -> Result<(), CompactRenderError> {
        let mut index = 0;
        let mut first_token = true;
        while index < ids.len() {
            let start = index;
            while index + 1 < ids.len() && ids[index].checked_add(1) == Some(ids[index + 1]) {
                index += 1;
            }
            let run_len = index - start + 1;
            if run_len >= 3 {
                write_list_separator(&mut self.writer, &mut first_token)?;
                write!(self.writer, "{}-{}", ids[start], ids[index])?;
            } else {
                for id in &ids[start..=index] {
                    write_list_separator(&mut self.writer, &mut first_token)?;
                    write!(self.writer, "{id}")?;
                }
            }
            index += 1;
        }
        Ok(())
    }

    fn current_frametime(&self) -> u64 {
        self.current_time
            .expect("an event time is set before facts")
    }

    fn validate_envelope(&mut self, event: &CloudEvent) -> Result<(), CompactRenderError> {
        if let Some(source) = self.source.as_deref() {
            if event.source != source {
                return Err(CompactRenderError::Source);
            }
            let Some(expected) = self.expected_sequence else {
                return Err(CompactRenderError::SequenceOverflow(u64::MAX));
            };
            if event.sequence != expected {
                return Err(CompactRenderError::Sequence {
                    expected,
                    actual: event.sequence,
                });
            }
        } else {
            self.source = Some(event.source.clone());
        }
        self.expected_sequence = event.sequence.checked_add(1);
        Ok(())
    }

    fn write_fact_header(&mut self) -> Result<(), CompactRenderError> {
        if !self.time_header_written {
            writeln!(
                self.writer,
                "@{}",
                self.current_time
                    .expect("an event time is set before facts")
            )?;
            self.time_header_written = true;
        }
        Ok(())
    }
}

fn write_list_separator<W: Write>(writer: &mut W, first_token: &mut bool) -> Result<(), io::Error> {
    if !*first_token {
        write!(writer, ",")?;
    }
    *first_token = false;
    Ok(())
}

fn validate_body_ids(body: &EventBody) -> Result<(), CompactRenderError> {
    match body {
        EventBody::Snapshot(snapshot) => {
            for element in &snapshot.elements {
                projection_number(&element.id)?;
            }
        }
        EventBody::ElementAppeared(data) => {
            projection_number(&data.element.id)?;
        }
        EventBody::ElementContentChanged(data) => {
            projection_number(&data.element.id)?;
        }
        EventBody::ElementMoved(data) => {
            projection_number(&data.id)?;
        }
        EventBody::ElementDisappeared(data) => {
            projection_number(&data.id)?;
        }
        EventBody::ScreenResized(_) => {}
        EventBody::ObservedActivity(data) => {
            if let crate::ObservedActivityKind::ElementRegionChanged { element_id } = &data.kind {
                projection_number(element_id)?;
            }
        }
    }
    Ok(())
}

fn projection_number(id: &str) -> Result<u64, CompactRenderError> {
    let Some(digits) = id.strip_prefix('e') else {
        return Err(CompactRenderError::ElementId(id.to_owned()));
    };
    if digits.len() < 6 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(CompactRenderError::ElementId(id.to_owned()));
    }
    let number = digits
        .parse::<u64>()
        .map_err(|_| CompactRenderError::ElementId(id.to_owned()))?;
    if number == 0 || format!("e{number:06}") != id {
        return Err(CompactRenderError::ElementId(id.to_owned()));
    }
    Ok(number)
}

fn meaningful_move(from_x: u32, from_y: u32, to_x: u32, to_y: u32) -> bool {
    from_x.abs_diff(to_x) >= MEANINGFUL_MOVE_PX || from_y.abs_diff(to_y) >= MEANINGFUL_MOVE_PX
}

fn quote(text: &str) -> String {
    serde_json::to_string(text).expect("serializing a string cannot fail")
}

fn state_text(text: &str) -> String {
    if text.is_empty() {
        return r"\e".to_owned();
    }
    let quoted = quote(text);
    quoted[1..quoted.len() - 1].to_owned()
}
