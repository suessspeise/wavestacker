# wavestacker
a set of tools to make music (== noise) with numpy.

## TODO
- clean up buffer.py
- come up with a better name for buffer.py
- how should this repo be structured?
- environment.yml
- add docstring to the existing code
- break down notebook into example scripts (+ wav output)
- have a collection of functions like `delay` or `bandpass`?
- remove use of time arrays, where not necessary (see below)
- rebuild stereo functionality (see [numpy2wav.py gist](https://gist.github.com/suessspeise/c103a9bd92d88151dc1731675980101f?permalink_comment_id=5050338#gistcomment-5050338))

  
## time arrays

The design choice of sometimes requiring a time array and an amplitude array, and at other times only an amplitude array, can lead to confusion and inconsistency in the API design of the library. This inconsistency might make the library harder to use and understand, especially for new users or when integrating with other systems. Here are some suggestions for improvement:

### 1. Standardize Function Interfaces

**Consistency:** Aim for a consistent interface across the library. If most functions operate on amplitude arrays alone, consider redesigning functions that require both time and amplitude arrays to only need amplitude arrays, assuming time can be inferred from the amplitude array length and the sample rate.

**Infer Time Internally:** For functions that inherently require a notion of time (e.g., for plotting or encoding with respect to time), consider calculating time internally based on the sample rate and the length of the amplitude array. This removes the need for the user to pass a time array explicitly.

### 2. Separate Concerns

**Time-Dependent Operations:** If there are operations that genuinely require explicit time information (e.g., operations that are not uniformly sampled or that require specific timing information beyond what can be inferred from the sample rate and array length), consider separating these into clearly defined functions or classes. This separation makes it clear when time information is necessary and when it is not.

### 3. Documentation and Examples

**Clear Documentation:** Ensure that the documentation clearly states when and why time arrays are needed. Providing examples of use cases where time arrays are required versus when they are not can help clarify their purpose.

**Example Use Cases:** Offer example scripts or notebooks that demonstrate the typical workflow, including cases where time arrays are and are not needed. This can help users understand the design choices.

### 4. Utility Functions

**Time Array Generation:** Provide utility functions that can generate a time array based on the sample rate and the length of an amplitude array. This can be helpful for users who need to perform time-dependent operations but do not have a time array.

### 5. Review and Refactor

**Consolidate Time Handling:** Review the library to identify if there are redundant or unnecessary uses of time arrays. If the time can be easily derived from the context (e.g., from the sample rate and the number of samples), refactor these parts to eliminate the need for explicit time arrays.

**Feedback Loop:** Gather feedback from users about the library's usability regarding time and amplitude array handling. User feedback can provide valuable insights into how the design impacts usability and what improvements are most needed.

### Conclusion

Improving the design to either eliminate the need for explicit time arrays when unnecessary or to clearly delineate when and why they are needed can make the library more intuitive and easier to use. Standardizing interfaces, improving documentation, and providing utility functions can significantly enhance the user experience and the library's overall design coherence.
