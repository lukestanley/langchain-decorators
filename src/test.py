# pylint: disable=too-few-public-methods,unused-argument,line-too-long,no-name-in-module
from typing import List
import logging

from pydantic import BaseModel, Field

from langchain.chat_models import ChatAnthropic
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_decorators import llm_prompt, PromptTypes, GlobalSettings

# logging.basicConfig(level=logging.CRITICAL)

TEMPERATURE = 0.3
FAST_CLAUDE_MODEL = "claude-instant-1"
BEST_CLAUDE_MODEL = "claude-1.3"


default_llm = ChatAnthropic(
    temperature=TEMPERATURE,
    model=FAST_CLAUDE_MODEL, 
    streaming=False
)

improver_llm = ChatAnthropic(
    temperature=0.7,
    model=BEST_CLAUDE_MODEL,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

GlobalSettings.define_settings(
    default_llm=default_llm,
    default_streaming_llm=default_llm,
)

PromptTypes.AGENT_REASONING.llm = default_llm


class SuggestedTextChange(BaseModel):
    """A rephrased version of an input text."""
    text: str = Field(description="Improved text")
    


class ScoreResult(BaseModel):
    """Metrics to evaluate the fidelity of a given text and how well received it might be."""
    faithfulness: float = Field(description="Faithfulness score from 0 to 1")
    spiciness: float = Field(
        description="Spiciness score, how spicy the text is from 0 to 1")
    offensiveness: float = Field(
        description="Offensive score, how offensive might the reader find the text, from 0 to 1")


class Attempt(BaseModel):
    """Stores attributes related to an attempt to rewrite a text."""
    attempt_summary: str
    improved_text: str
    scores: ScoreResult
    overall_score: float
    feedback: str


class OriginalScoreSet(BaseModel):
    """A score set evaluating the spiciness of a given text."""
    review: str = Field(description="A review the original text")
    spiciness: float = Field(
        description="Spiciness score, how spicy the text is from 0 to 1")
    offensiveness: float = Field(
        description="Offensive score, how offensive might the reader find the text, from 0 to 1")


@llm_prompt()
def score_original_text(original_text: str) -> OriginalScoreSet:
    """
    We want to make the internet a more calm and constructive place, so it is important to evaluate text.
    Here is the original text:
    `{original_text}`

    Here is a guide to the scoring the text:
    A spiciness score of 0 is calm ideal.
    A spiciness score of 1 is the worst, very harsh text that almost makes the reader feel attacked.

    Could the text be considered offensive, or inflammatory?
    Assume people are very easily offended, they might even be depressed.
    Let's think step by step about how the text might be interpreted as offensive.

    {FORMAT_INSTRUCTIONS}
    """
    return


@llm_prompt(llm=improver_llm)
def improve_text(original_text: str, prior_attempt_summary: str, original_text_review: str) -> SuggestedTextChange:
    """
    Rephrasing messages to be more calm is an important part of reducing possible harm.
    Make the text calm, unoffensive and kind while following the writer's intent.
    Here is the original text:
    {original_text}

    Your task is to rephrase possibly unkind text, to ensure it is calm and constructive, without changing the intended meaning.
    The improved text should have a softened tone, avoiding judgemental and extreme words.
    Make sure the refined text is a good reflection of the original text, without adding new ideas.
    1. Rather than accusations, share perspective.
    2. Remove or soften judgemental language.
    3. Focus on specific actions rather than character.
    4. Rephrase extreme words like "always", "never" or "everyone" to be more moderate.
    5. Focus on softening the tone, rather than changing the substance or meaning.
    6. Use gentler alternatives to express similar points.
    7. Ensure the new text could not be interpreted as offensive, avoid presuming about the reader.
    8. Avoid being patronising, condescending or preachy.
    9. Never add completely new ideas, ONLY build upon what's already there.
    10. For example, you might reframe an existing point to be more balanced. Never introduce unrelated concepts.

    Here is a review of the original text:
    {original_text_review}

    {prior_attempt_summary}

    Output the improved text in JSON format, with the key "text" and no remarks.
    {FORMAT_INSTRUCTIONS}
    """
    return


@llm_prompt()
def score_text_edit(original_text: str, improved_text: str) -> ScoreResult:
    """
    We want to make the internet a more calm and constructive place, so it is important to evaluate attempts at improving text.
    Here is the original text:
    `{original_text}`

    This is the edited text: 
    `{improved_text}`

    Here is a guide to the scoring the new text:
    A spiciness score of 0 is calm ideal.
    A spiciness score of 1 is the worst, very harsh text that almost makes the reader feel attacked.

    A faithfulness score of 1 would have the same semantic intent as the original text.
    A faithfulness score of 0 would mean the text has lost all semantic similarity.

    Only score the edited text, not the original text.

    {FORMAT_INSTRUCTIONS}
    """
    return


class TextReview(BaseModel):
    """A review of a given text."""
    review: str = Field(description="A review of the text")


@llm_prompt()
def review_new_text(original_text: str, improved_text: str) -> TextReview:
    """A potentially inflammatory message was rewrittern.
    How well does the rewrite reflect the original intent?
    Could the new text may be considered offensive, or inflammatory?
    Let's think step-by-step about how the new text might be interpreted as offensive.
    Assume people are very easily offended, they might even be depressed.

    Original text:
    {original_text}

    Improved text:
    {improved_text}

    Be detailed but concise, use abreiviations. To be concise, do not quote the text.
    Ensure you output a valid JSON object with the key of review and no remarks.
    {FORMAT_INSTRUCTIONS}
    """
    return


@llm_prompt()
def condense_review(original_text: str, text_review: str) -> TextReview:
    """
    A review of text was produced, but it is too long.
    Here is the original text, that the review was written about:
    {original_text}
    Here is the review that is too long:
    {text_review}
    Please condense the review to be shorter, while still covering the same points.

    {FORMAT_INSTRUCTIONS}
    """
    return


def calculate_overall_score(score_set: ScoreResult):
    """Calculate a score for a given text by balancing how true it stays to the original text, how calm it is, and how non-offensive it is.

    Args:
        score_set (ScoreResult): The scores of faithfulness, spiciness, and offensiveness of a text edit.

    Returns:
        float: A score between 0 and 1, with 1 being the best possible score and 0 being the worst.
    """

    calmness = 1 - (score_set.spiciness * 2)

    # We magnify the importance of not being offensive by multiplying by offensiveness by 2:
    non_offensiveness = 1 - (score_set.offensiveness * 4)

    # Calculate the overall score as the average of faithfulness, calmness, and non_offensiveness
    score = (score_set.faithfulness + calmness + non_offensiveness) / 3

    # Make sure the score is between 0 and 1
    score = max(0, min(1, score))

    return score


def generate_prior_attempt_summary(past_attempts, best_attempt_limit=2) -> str:
    """Makes a text summary of the best prior attempts."""

    # Return an new line if there are no prior attempts:
    if not past_attempts:
        return "\n"

    # Otherwise, we have some prior attempts to summarise:
    prior_attempt_summary = "Here are the previous attempts:\n\n"
    past_attempts.sort(key=lambda x: x.overall_score, reverse=True)
    top_attempts = past_attempts[:best_attempt_limit]

    for attempt in top_attempts:
        prior_attempt_summary += attempt.attempt_summary + "\n\n"

    logging.info("prior_attempt_summary: %s", prior_attempt_summary)
    return prior_attempt_summary


def evaluate_new_attempt(original_text, improved):
    """Evaluate a new attempt to improve the text and return an Attempt object."""
    feedback: TextReview = review_new_text(original_text=original_text,
                                           improved_text=improved.text)
    score_set: ScoreResult = score_text_edit(original_text=original_text,
                                             improved_text=improved.text)
    overall_score: float = calculate_overall_score(score_set)
    attempt_summary = f"""
        Previous text refinement attempt text: {improved.text}
        Review of suggested refinement: {feedback.review}
        Refinement score: {overall_score:.2f}
        """

    new_attempt = Attempt(
        attempt_summary=attempt_summary,
        improved_text=improved.text,
        scores=score_set,
        overall_score=overall_score,
        feedback=feedback.review
    )

    return new_attempt


def get_best_text(original_text: str, rephrase_attempts_limit=4, good_overall_score=0.85):
    """Get the best text improvement, hill climbing style."""
    best_attempt: Attempt = None
    graded_text_rephrasings: List[Attempt] = []
    rephrase_attempts = 0

    original_text_benchmark: OriginalScoreSet = score_original_text(
        original_text=original_text)
    logging.info("Original text benchmark: %s",
                 original_text_benchmark)
    original_text_review_summary: str = condense_review(
        original_text=original_text, text_review=original_text_benchmark.review).review

    logging.info("original_text_review_summary: %s",
                 original_text_review_summary)
    while rephrase_attempts < rephrase_attempts_limit:

        prior_attempt_summary: str = generate_prior_attempt_summary(
            graded_text_rephrasings)

        new_text_suggestion = improve_text(
            original_text=original_text,
            prior_attempt_summary=prior_attempt_summary,
            original_text_review=original_text_review_summary
        )

        new_attempt = evaluate_new_attempt(original_text, new_text_suggestion)
        logging.debug("New attempt: %s", new_attempt)

        graded_text_rephrasings.append(new_attempt)

        rephrase_attempts += 1

        # Update the best attempt
        if not best_attempt or new_attempt.overall_score > best_attempt.overall_score:
            best_attempt = new_attempt

        if new_attempt.overall_score > good_overall_score:
            break
    logging.info("Best improved text: %s", best_attempt)

    return best_attempt.improved_text


SPICY_COMMENT_TO_REPHRASE = """Stop chasing dreams instead. Life is not a Hollywood movie. Not everyone is going to get to be a famous billionaire. Adjust your expectations to reality, and stop thinking so highly of yourself, stop judging others. Assume the responsibility for the things that happen in your life. It is kind of annoying to read your text, it is always some external thing that "happened" to you, and it is always other people who are not up to your standards. At some moment you even declare with despair. And guess what? This is true and false at the same time, in a fundamental level most people are not remarkable, and you probably aren't too. But at the same time, nobody is the same, you have worth just by being, and other people have too. The impression I get is that you must be someone incredibly annoying to work with, and that your performance is not even nearly close to what you think it is, and that you really need to come down to earth. Stop looking outside, work on yourself instead. You'll never be satisfied just by changing jobs. Do therapy if you wish, become acquainted with stoicism, be a volunteer in some poor country, whatever, but do something to regain control of your life, to get some perspective, and to adjust your expectations to reality."""
# From elzbardico on https://news.ycombinator.com/item?id=36119858
print(get_best_text(SPICY_COMMENT_TO_REPHRASE))
