#[derive(Clone, PartialEq, Debug, Default)]
pub enum TextConcatenator {
    DoubleNewline,
    #[default]
    SingleNewline,
    Space,
    Comma,
    Custom(String),
}

impl TextConcatenator {
    pub fn as_str(&self) -> &str {
        match self {
            TextConcatenator::DoubleNewline => "\n\n",
            TextConcatenator::SingleNewline => "\n",
            TextConcatenator::Space => " ",
            TextConcatenator::Comma => ", ",
            TextConcatenator::Custom(custom) => custom,
        }
    }
}

pub trait TextConcatenatorTrait {
    fn concatenator_mut(&mut self) -> &mut TextConcatenator;

    fn clear_built(&self);

    fn concate_deol(&mut self) -> &mut Self {
        if self.concatenator_mut() != &TextConcatenator::DoubleNewline {
            *self.concatenator_mut() = TextConcatenator::DoubleNewline;
            self.clear_built();
        }
        self
    }

    fn concate_seol(&mut self) -> &mut Self {
        if self.concatenator_mut() != &TextConcatenator::SingleNewline {
            *self.concatenator_mut() = TextConcatenator::SingleNewline;
            self.clear_built();
        }
        self
    }

    fn concate_space(&mut self) -> &mut Self {
        if self.concatenator_mut() != &TextConcatenator::Space {
            *self.concatenator_mut() = TextConcatenator::Space;
            self.clear_built();
        }
        self
    }

    fn concate_comma(&mut self) -> &mut Self {
        if self.concatenator_mut() != &TextConcatenator::Comma {
            *self.concatenator_mut() = TextConcatenator::Comma;
            self.clear_built();
        }
        self
    }

    fn concate_custom<T: AsRef<str>>(&mut self, custom: T) -> &mut Self {
        if self.concatenator_mut() != &TextConcatenator::Custom(custom.as_ref().to_owned()) {
            *self.concatenator_mut() = TextConcatenator::Custom(custom.as_ref().to_owned());
            self.clear_built();
        }
        self
    }
}
