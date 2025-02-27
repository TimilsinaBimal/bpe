import regex as re


class RegexTokenizer:
    def match_pattern(self, text: str, model: str) -> list:
        pattern = self._get_pattern(model)
        matches = re.findall(pattern, text)
        return matches

    def _get_pattern(self, model: str) -> str | None:
        pattern_mapping = {
            "gpt4": self._gpt4(),
        }
        return pattern_mapping.get(model)

    @staticmethod
    def _gpt4():
        contractions = r"""'(?i:[sdmt]|ll|ve|re)"""
        # matches contractions should've, It's and separate them to [should 've]
        # 's, 'd, 'm, 't, 'll, 've, 're
        words = r"""[^\r\n\p{L}\p{N}]?+\p{L}++"""  # matches all words
        numbers = r"""\p{N}{1,3}+"""  # matches numbers from one to 3 digits
        special_characters = r""" ?[^\s\p{L}\p{N}]++[\r\n]*+"""
        # handle special characters, punctuations even if they are at start, end or
        # without spaces with spaces
        spaces = "".join(
            [
                r"\s++$",  # handle spaces at the end of the line
                r"|\s*[\r\n]",  # handle new lines
                r"|\s+(?!\S)",  # handle spaces that are not followed by a word
                r"|\s",  # handle spaces
            ]
        )

        return f"{contractions}|{words}|{numbers}|{special_characters}|{spaces}"
