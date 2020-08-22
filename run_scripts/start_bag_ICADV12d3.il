;; SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
;; Copyright 2018 Regents of the University of California
;; All rights reserved.
;;
;; Redistribution and use in source and binary forms, with or without
;; modification, are permitted provided that the following conditions are met:
;;
;; * Redistributions of source code must retain the above copyright notice, this
;;   list of conditions and the following disclaimer.
;;
;; * Redistributions in binary form must reproduce the above copyright notice,
;;   this list of conditions and the following disclaimer in the documentation
;;   and/or other materials provided with the distribution.
;;
;; * Neither the name of the copyright holder nor the names of its
;;   contributors may be used to endorse or promote products derived from
;;   this software without specific prior written permission.
;;
;; THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
;; AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
;; IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
;; DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
;; FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
;; DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
;; SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
;; CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
;; OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
;; OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

;; Copyright 2019 Blue Cheetah Analog Design Inc.
;;
;; Licensed under the Apache License, Version 2.0 (the "License");
;; you may not use this file except in compliance with the License.
;; You may obtain a copy of the License at
;;
;;     http://www.apache.org/licenses/LICENSE-2.0
;;
;; Unless required by applicable law or agreed to in writing, software
;; distributed under the License is distributed on an "AS IS" BASIS,
;; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
;; See the License for the specific language governing permissions and
;; limitations under the License.

/*  Note:

Due to licensing reasons, this skill script is missing the function
CCSinvokeCdfCallbacks() from Cadence solution 11018344, which executes
CDF parameters callback from skill.

If you do not need to instantiate a pcell instance, this method
is not needed.

Eric Chang, Mar 2, 2017.

*/


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Virtuoso Database operations functions  ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; reads a skill data structure from file
procedure( parse_data_from_file( fname "t" )
    let( (p ans)
        unless( p = infile( fname )
            error("Cannot open file %s" fname)
        )
        ans = parse_data_from_file_helper(p)
        close( p )
        ans
    )
)

; recursive helper for parse_data_from_file
procedure( parse_data_from_file_helper( p )
    let( (line item ans finish key)
        gets( line p )
        ; remove newline
        line = substring(line 1 strlen(line) - 1)
        ; printf("read line: %s\n" line)
        cond(
            (line == "#list"
            ; parse a list
            ans = tconc(nil 0)
            while( nequal(item = parse_data_from_file_helper(p) "#end")
                tconc(ans item)
            )
            ; printf("returning list ")
            ; print(cdar(ans))
            ; printf("\n")
            cdar(ans)
            )
            (line == "#prop_list"
            ; parse a disembodied property list
            ans = ncons(nil)
            finish = nil
            while( !finish
                key = parse_data_from_file_helper(p)
                if( key == "#end" then
                    finish = 't
                else
                    item = parse_data_from_file_helper(p)
                    putprop(ans item key)
                )
            )
            ans
            )
            ; parse a float
            (strncmp( line "#float" 6 ) == 0
                cdfParseFloatString(cadr(parseString(line)))
            )
            ; parse an int
            (strncmp( line "#int" 4 ) == 0
                atoi(cadr(parseString(line)))
            )
            ; parse a boolean
            (strncmp( line "#bool" 5 ) == 0
                if( atoi(cadr(parseString(line))) == 1 then
                    't
                else
                    nil
                )
            )
            ; parse a string token or #end
            ('t
                ; printf("returning str %s\n" line)
                line
            )
        )
    )
)


; close all opened cellviews
procedure( close_all_cellviews()
    foreach( cv dbGetOpenCellViews()
        dbPurge(cv)
    )
    't
)


; release write locks from all the given cellviews
procedure( release_write_locks( lib_name cell_view_list_f "tt" )
    let( (cell_view_list lib_obj cv)
        cell_view_list = parse_data_from_file(cell_view_list_f)
        when( lib_obj = ddGetObj(lib_name nil nil nil nil "r")
            foreach( info cell_view_list
                when( cv = dbFindOpenCellView( lib_obj car(info) cadr(info) )
                    dbReopen(cv, "r")
                    dbClose(cv)
                )
            )
            ddReleaseObj(lib_obj)
        )
        t
    )
)


; refresh all given cell views
procedure( refresh_cellviews( lib_name cell_view_list_f "tt" )
    let( (cell_view_list lib_obj cv)
        cell_view_list = parse_data_from_file(cell_view_list_f)
        when( lib_obj = ddGetObj(lib_name nil nil nil nil "r")
            foreach( info cell_view_list
                when( cv = dbFindOpenCellView( lib_obj car(info) cadr(info) )
                    dbRefreshCellView(cv)
                )
            )
            ddReleaseObj(lib_obj)
        )
        t
    )
)

; modify label text in a layout
procedure( modify_label(lib_name cell_name reg_before reg_after "tttt")
    let((cv lbl_txt new_lbl)
        unless( cv = dbOpenCellViewByType( lib_name cell_name "layout" nil "a" )
            error( "Cannot find layout view of cell %s__%s" lib_name cell_name )
        )

        foreach(shape_obj cv~>shapes
            when( lbl_txt = shape_obj->theLabel
                when( rexMatchp(reg_before lbl_txt)
                    new_lbl = rexReplace(lbl_txt reg_after 1)
                    shape_obj->theLabel = new_lbl
                )
            )
        )

        dbSave(cv)
        dbClose(cv)
    )
)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  BAG server related functions            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

procedure( stdoutHandler(ipcId data)
    let( (result result_str)
        if( bag_server_started > 0 then
            printf("*INFO* Evaluate expression from BAG process: %s\n" data)
            if( result = errsetstring(data 't) then
                sprintf(result_str "%A\n" car(result))
            else
                sprintf(result_str "%s\n" car(nthelem(5 errset.errset)))
            )
            printf("*INFO* Sending result to BAG process: %s" result_str)
            ipcWriteProcess(ipcId sprintf(nil "%d\n" strlen(result_str)))
            ipcWriteProcess(ipcId result_str)
            't
        else
            if( data == "BAG skill server has started.  Yay!\n" then
                bag_server_started = 1
                printf("*INFO* BAG skill server started.\n")
            else
                printf("*INFO* Waiting for BAG skill server.  Message: %s\n" data)
            )
        )
    )
)

procedure( stderrHandler(ipcId data)
    warn("BAG server process error: %s\n" data)
    warn("Shutting down BAG server.")
    ipcKillProcess(ipcId)
    't
)

procedure( exitHandler(ipcId exitId)
    printf("*INFO* BAG server process exited with status: %d\n" exitId)
    't
)

procedure( start_bag_server()
    bag_server_started = 0
    printf("*INFO* Starting BAG server process.\n")
    ipcBeginProcess("bash virt_server.sh" "" 'stdoutHandler 'stderrHandler 'exitHandler "")
)

bag_server_started = 0
bag_modify_adexl_counter = 0
bag_proc = start_bag_server()
