module PyPlotExt

#=  Plotting for the monogenic layers  =#

using MonogenicFilterFlux
using PyPlot

# output: the output of the first Layer
# index: the index of the dataset
function MonogenicFilterFlux.visual_first_layer(output, index=1)

    if length(size(output)) == 4

        nchannel = 3
        scale = Int(size(output, 3) / nchannel)

        fig = PyPlot.figure(figsize=(5, 10))

        local p

        for scal = scale:-1:1

            o1 = output[:, :, 1+3*(scal-1), index]
            o2 = output[:, :, 2+3*(scal-1), index]
            o3 = output[:, :, 3+3*(scal-1), index]

            if scal == 1
                PyPlot.subplot(scale, nchannel, 1 + 3 * (scal - 1), title="1",
                    ylabel=string(scal), xticks=[], yticks=[])
                p = PyPlot.imshow(o1, cmap="gray")
                PyPlot.subplot(scale, nchannel, 2 + 3 * (scal - 1), title="i")
                PyPlot.imshow(o2, cmap="gray")
                PyPlot.axis("off")
                PyPlot.subplot(scale, nchannel, 3 + 3 * (scal - 1), title="j")
                PyPlot.imshow(o3, cmap="gray")
                PyPlot.axis("off")
            else
                PyPlot.subplot(scale, nchannel, 1 + 3 * (scal - 1),
                    ylabel=string(scal), xticks=[], yticks=[])
                PyPlot.imshow(o1, cmap="gray")
                PyPlot.subplot(scale, nchannel, 2 + 3 * (scal - 1))
                PyPlot.imshow(o2, cmap="gray")
                PyPlot.axis("off")
                PyPlot.subplot(scale, nchannel, 3 + 3 * (scal - 1))
                PyPlot.imshow(o3, cmap="gray")
                PyPlot.axis("off")
            end

        end

        fig.text(0.05, 0.5, "scale", ha="center", va="center", rotation=90)
        PyPlot.suptitle("First Layer Output")
        fig.subplots_adjust(right=0.8)
        cbar = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        fig.colorbar(p, cax=cbar)

        return fig
    else
        print("Wrong Dimension. The input is 4 dimensional.\n")
    end

end

# output: the output of the second Layer
# index: the index of the dataset
function MonogenicFilterFlux.visual_second_layer(output, index=1)

    if length(size(output)) == 5

        nchannel = 3

        n_layer1 = size(output, 4)
        n_layer2 = size(output, 3)

        scale_layer1 = Int(n_layer1 / nchannel)
        scale_layer2 = Int(n_layer2 / nchannel)

        fig = PyPlot.figure(figsize=(10, 10))

        local p

        for scal2 = scale_layer2:-1:1
            for scal1 = scale_layer1:-1:1

                # index within the block
                for k = nchannel:-1:1 # row in block
                    for j = nchannel:-1:1 # column in block
                        o1 = output[:, :, k+nchannel*(scal2-1), j+nchannel*(scal1-1), index]

                        panel = n_layer2 * ((k - 1) + nchannel * (scal2 - 1)) + j + nchannel * (scal1 - 1)

                        # first row
                        bool_1st_row = false
                        for jj = 1:nchannel
                            if (scal2 == 1) & (k == 1) & (j == jj)

                                if jj == 1
                                    title_name = "(1,:)"
                                elseif jj == 2
                                    title_name = "(i,:)"
                                elseif jj == 3
                                    title_name = "(j,:)"
                                end

                                PyPlot.subplot(n_layer2, n_layer1, panel,
                                    xticks=[], yticks=[], title=title_name)
                                PyPlot.imshow(o1, cmap="gray")
                                bool_1st_row = true
                            end
                        end

                        # last column
                        bool_last_column = false
                        for jj = 1:nchannel
                            if (scal1 == scale_layer1) & (j == nchannel) & (k == jj)

                                if jj == 1
                                    title_name = "(:,1)"
                                elseif jj == 2
                                    title_name = "(:,i)"
                                elseif jj == 3
                                    title_name = "(:,j)"
                                end
                                pp = PyPlot.subplot(n_layer2, n_layer1, panel,
                                    xticks=[], yticks=[], ylabel=title_name)
                                pp.yaxis.set_label_position("right")
                                PyPlot.imshow(o1, cmap="gray")
                                bool_last_column = true
                            end
                        end

                        if (bool_last_column == false) & (bool_1st_row == false)
                            PyPlot.subplot(n_layer2, n_layer1, panel, xticks=[], yticks=[])
                            PyPlot.imshow(o1, cmap="gray")
                        end

                        # first column
                        if (scal1 == 1) & (j == 1)
                            PyPlot.subplot(n_layer2, n_layer1, panel,
                                xticks=[], yticks=[], ylabel=Int(scal2))
                            PyPlot.imshow(o1, cmap="gray")
                        end

                        # last row
                        if (scal2 == scale_layer2) & (k == nchannel)
                            PyPlot.subplot(n_layer2, n_layer1, panel,
                                xticks=[], yticks=[], xlabel=Int(scal1))
                            PyPlot.imshow(o1, cmap="gray")
                        end

                        # (1,1)
                        if (scal1 == 1) & (scal2 == 1) & (j == 1) & (k == 1)
                            PyPlot.subplot(n_layer2, n_layer1, panel,
                                xticks=[], yticks=[], title="(1,:)", ylabel=1)
                            p = PyPlot.imshow(o1, cmap="gray")
                        end

                        # (1,last)
                        if (scal1 == scale_layer1) & (scal2 == 1) & (j == nchannel) & (k == 1)
                            pp = PyPlot.subplot(n_layer2, n_layer1, panel,
                                xticks=[], yticks=[], title="(j,:)", ylabel="(:,1)")
                            pp.yaxis.set_label_position("right")
                            PyPlot.imshow(o1, cmap="gray")
                            bool_last_column = true
                        end

                        # (last,1)
                        if (scal1 == 1) & (scal2 == scale_layer2) & (j == 1) & (k == nchannel)
                            PyPlot.subplot(n_layer2, n_layer1, panel,
                                xticks=[], yticks=[], xlabel=1, ylabel=scal2)
                            PyPlot.imshow(o1, cmap="gray")
                        end

                        # (last,last)
                        if (scal1 == scale_layer1) & (scal2 == scale_layer2) & (j == nchannel) & (k == nchannel)
                            pp = PyPlot.subplot(n_layer2, n_layer1, panel,
                                xticks=[], yticks=[], xlabel=scal1, ylabel="(:,j)")
                            pp.yaxis.set_label_position("right")
                            PyPlot.imshow(o1, cmap="gray")
                            bool_last_column = true
                        end

                    end
                end

            end
        end

        fig.text(0.5, 0.05, "first layer index", ha="center", va="center")
        fig.text(0.08, 0.5, "second layer index", ha="center", va="center", rotation=90)
        PyPlot.suptitle("Second Layer Output")
        fig.subplots_adjust(right=0.8)
        cbar = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        fig.colorbar(p, cax=cbar)

        return fig
    else
        print("Wrong Dimension. The input is 5 dimensional.\n")
    end

end

end # module
